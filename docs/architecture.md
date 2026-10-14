# Architecture

```text
src/openkalolumen/
├── app.py                 CLI and Qt lifecycle
├── config.py              Resource paths independent of the working directory
├── errors.py              Cooperative cancellation exception
├── state.py               Layers, clipping, measurements, and scene schema
├── samples.py             Downloads, checksums, and provenance
├── imaging/
│   ├── volume.py          Image IO, DICOM, coordinates, and label alignment
│   ├── mpr.py             Physical-plane resampling
│   └── drr.py             CT attenuation integration
├── render/
│   ├── geometry.py        Meshes, finite materials, clipping, caches, and export
│   ├── materials.py       PBR and material-contact visibility
│   └── engine.py          VTK rendering, sections, cameras, and picking
├── desktop/
│   ├── window.py          State ownership, controls, and layer editing
│   ├── navigation.py      Clipping, slices, cameras, and DRR controls
│   ├── scenes.py          Scene restoration, export, and native capture
│   ├── segmentation.py    Model metadata and QProcess lifecycle
│   ├── tasks.py           Background execution and obsolete-result isolation
│   └── widgets.py         Image canvases and slice panels
└── segmentation/
    └── worker.py          VISTA3D inference in an isolated environment
```

`imaging` and `render` do not depend on desktop windows. `state` contains serializable state and parameter validation. The main window owns the active image and controls; navigation, scene, and segmentation mixins share its initialized state rather than creating separate windows or thread pools. The standalone segmentation worker communicates through JSON Lines and does not import desktop modules.

## Coordinates and source data

NumPy arrays use `z, y, x` ordering. Voxel indices, spacing, and physical transforms use `x, y, z`. External coordinates are LPS millimeters. VTK scalar grids use local axis-aligned coordinates and transform into the physical world using the image direction matrix. MPR, label alignment, picking, DRR, and STL/PLY export follow the same convention.

Original intensities define geometric extraction and MPR. Display filtering specifies a Gaussian standard deviation in millimeters and affects only the 3D display field. Measurements require filtering to be disabled; export extracts original scalar geometry again. ROI masks and labels control visibility without modifying source data.

## Rendering paths

GPU previews support threshold interaction. Detailed meshes provide stable rendering, measurements, and export. Volume rendering shows intensity intervals throughout the volume. Each path has a distinct responsibility.

Interval surfaces represent lower and upper scalar boundaries with reversed upper-boundary normals. Finite material intervals combine intensities and ROI masks into closed materials, retaining tissue thickness and cavity openings in sections. These remain separate explicit layer types without automatic conversion.

Material colors follow source-boundary identity rather than camera orientation. Internal contacts between opaque materials are hidden, while complete closed meshes remain cached. Clipped export validates closure and nonmanifold edges. Numerical epsilon and vertex welding address numerical coincidences only.

## Background work and persistence

Tasks are keyed and versioned. Replacing a task invalidates older results, even if an old thread finishes later. Inference has its own process lifecycle and cache; cache identities include image and weight content, class IDs, requested device, dependency versions, and preprocessing version.

Scene schema version 1 stores resource paths and image fingerprints without embedding images. Relocated resources must be available at the recorded paths. Writable caches belong to `AppPaths.data`, outside the installed package and application bundle.

## Maintenance boundaries

Public CLI entry points are `openkalolumen` and `openkalolumen-samples`. Validation uses pytest and the synthetic desktop smoke test. Startup uses synthetic data unless an image or scene is explicitly selected. Source packages exclude local scenes, downloaded images, weights, reference repositories, and personal configuration.

Segmentation prompts use IDs from model metadata directly. Resource paths use explicit CLI options, environment variables, or platform defaults. The application does not discover models by searching a developer checkout or rely on presets tied to particular images and local outputs.
