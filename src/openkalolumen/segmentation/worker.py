"""Isolated VISTA3D inference. JSON lines are the desktop process protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def emit(event, **fields):
    """Write one flushed JSON event to the desktop process protocol."""
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default="115,6")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--roi", default="192,192,128")
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        import monai
        import numpy as np
        import SimpleITK as sitk
        import torch
        from monai.data import MetaTensor
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import vista3d132
        from monai.transforms import (
            CastToTyped,
            Compose,
            CropForegroundd,
            EnsureChannelFirstd,
            EnsureTyped,
            Invertd,
            LoadImaged,
            Orientationd,
            ScaleIntensityRangePercentilesd,
            Spacingd,
        )

        classes = sorted(set(int(x) for x in args.classes.split(",")))
        roi = tuple(int(x) for x in args.roi.split(","))
        if not classes or any(x < 1 or x >= 512 for x in classes):
            raise ValueError("Class IDs are outside the model prompt embedding range.")
        if len(roi) != 3 or any(x < 32 or x % 16 for x in roi):
            raise ValueError(
                "The window must have three dimensions, each a multiple of 16 and at least 32."
            )
        if not 0 <= args.overlap < 1:
            raise ValueError("Window overlap must be in [0, 1).")
        model_dir = Path(args.model)
        definitions = json.loads((model_dir / "metadata.json").read_text())[
            "network_data_format"
        ]["outputs"]["pred"]["channel_def"]
        if any(
            str(c) not in definitions or "deprecated" in definitions[str(c)].lower()
            for c in classes
        ):
            raise ValueError("A requested class does not exist or is deprecated.")
        weights = model_dir / "vista3d_pretrained_model" / "model.pt"
        if not weights.exists():
            raise FileNotFoundError(weights)
        h = hashlib.sha256()
        for path in (Path(args.image), weights):
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    h.update(block)
        settings = {
            "classes": classes,
            "roi": roi,
            "overlap": args.overlap,
            "device_request": args.device,
            "precision": "float32",
            "spacing": [1.5] * 3,
            "preprocess_version": 3,
            "torch": torch.__version__,
            "monai": monai.__version__,
        }
        h.update(json.dumps(settings, sort_keys=True).encode())
        folder = Path(args.output).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / (h.hexdigest() + ".labels.nii.gz")
        if output.exists() and output.with_suffix(".json").exists():
            cached = json.loads(output.with_suffix(".json").read_text())
            if not any(cached.get("voxel_counts", {}).values()):
                raise ValueError(
                    "The model did not identify any requested organ. This partial or pediatric CT may be outside its supported domain; use existing labels or a manual ROI."
                )
            emit("result", path=str(output), cached=True)
            return 0
        emit("progress", percent=2, message="Loading local VISTA3D weights")
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        network = vista3d132(encoder_embed_dim=48, in_channels=1).eval()
        network.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        device = "cpu"
        fallback = ""
        probe_difference = None
        if args.device != "cpu":
            if torch.backends.mps.is_available():
                generator = torch.Generator().manual_seed(7)
                probe = torch.rand((1, 1, 32, 32, 32), generator=generator)
                cls = torch.tensor([[classes[0]]], dtype=torch.long)
                with torch.inference_mode():
                    reference = network(probe, class_vector=cls, transpose=True).cpu()
                    try:
                        network.to("mps")
                        accelerated = network(
                            probe.to("mps"), class_vector=cls.to("mps"), transpose=True
                        ).cpu()
                        torch.testing.assert_close(
                            accelerated, reference, rtol=0.02, atol=0.003
                        )
                        probe_difference = float((accelerated - reference).abs().max())
                        device = "mps"
                    except Exception as e:
                        fallback = f"MPS validation failed; using CPU: {e}"
                        network.to("cpu")
                        torch.mps.empty_cache()
                        if args.device == "mps":
                            raise
            elif args.device == "mps":
                raise RuntimeError("MPS is unavailable on this device.")
        emit(
            "progress",
            percent=8,
            message=fallback
            or f"Using {device.upper()}; resampling and normalizing intensities",
        )

        pre = Compose(
            [
                LoadImaged(keys="image", image_only=True),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys="image", device="cpu", track_meta=True),
                Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
                CropForegroundd(
                    keys="image", source_key="image", margin=10, allow_smaller=True
                ),
                ScaleIntensityRangePercentilesd(
                    keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True
                ),
                Orientationd(keys="image", axcodes="RAS", labels=None),
                CastToTyped(keys="image", dtype=torch.float32),
            ]
        )
        data = pre({"image": str(Path(args.image).resolve())})
        image = data["image"].unsqueeze(0)
        class_vector = torch.tensor(classes, dtype=torch.long).reshape(-1, 1)
        calls = 0

        import math

        total = math.prod(
            max(1, math.ceil((max(n, r) - r) / max(int(r * (1 - args.overlap)), 1)) + 1)
            for n, r in zip(image.shape[2:], roi)
        )

        def predictor(patch):
            nonlocal calls
            calls += 1
            emit(
                "progress",
                percent=10 + int(78 * min(calls, total) / total),
                message=f"{device.upper()} segmentation window {calls}/{total}",
            )

            outputs = [
                network(patch, class_vector=chunk.to(patch.device), transpose=True)
                for chunk in class_vector.split(4)
            ]
            return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

        def infer():
            nonlocal calls
            calls = 0
            with torch.inference_mode():
                return sliding_window_inference(
                    image,
                    roi,
                    1,
                    predictor,
                    overlap=args.overlap,
                    mode="gaussian",
                    padding_mode="replicate",
                    sw_device=device,
                    device="cpu",
                )

        try:
            logits = infer()
        except RuntimeError as e:
            if device != "mps" or args.device != "auto":
                raise
            fallback = f"MPS inference failed; retrying on CPU: {e}"
            emit("progress", percent=10, message=fallback)
            device = "cpu"
            network.to("cpu")
            torch.mps.empty_cache()
            logits = infer()
        emit(
            "progress",
            percent=90,
            message="Restoring the original image coordinates and voxel grid",
        )
        scores, indices = (
            logits[0].as_tensor().max(dim=0)
            if hasattr(logits, "as_tensor")
            else logits[0].max(dim=0)
        )
        mapping = torch.tensor(classes, dtype=torch.int16)
        labels = torch.where(scores > 0, mapping[indices], 0).unsqueeze(0)
        data["pred"] = MetaTensor(labels)
        restored = Invertd(
            keys="pred",
            transform=pre,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        )(data)["pred"]
        label_xyz = restored[0].cpu().numpy().astype(np.uint16)
        if not np.any(label_xyz):
            raise ValueError(
                "The model did not identify any requested organ. Use existing labels or a manual ROI; empty segmentation was not saved."
            )
        original = sitk.ReadImage(str(args.image))
        result = sitk.GetImageFromArray(np.transpose(label_xyz, (2, 1, 0)))
        if result.GetSize() != original.GetSize():
            raise RuntimeError(
                "The inverse-transformed result has a different image size."
            )
        result.CopyInformation(original)
        temporary = output.with_name(output.name.replace(".nii.gz", ".partial.nii.gz"))
        sitk.WriteImage(result, str(temporary), True)
        temporary.replace(output)
        counts = {str(c): int((label_xyz == c).sum()) for c in classes}
        metadata = {
            **settings,
            "device": device,
            "fallback": fallback,
            "source": str(Path(args.image).resolve()),
            "mps_cpu_probe_max_logit_difference": probe_difference,
            "model": str(weights.resolve()),
            "cache_key": h.hexdigest(),
            "voxel_counts": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "coordinate_system": "LPS",
            "geometry": {
                "size": original.GetSize(),
                "spacing": original.GetSpacing(),
                "origin": original.GetOrigin(),
                "direction": original.GetDirection(),
            },
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        emit("progress", percent=100, message="Organ segmentation completed")
        emit("result", path=str(output), cached=False, metadata=metadata)
        return 0
    except Exception as e:
        emit("error", message=str(e))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
