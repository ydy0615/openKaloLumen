import numpy as np
import pytest
import SimpleITK as sitk

from openkalolumen.imaging.volume import dicom_series, read_dicom


def write_series(folder, gap=False):
    values = np.full((6, 8, 9), -900, dtype=np.int16)
    values[:, 2:6, 3:7] = 100
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.2, 1.5, 2.0))
    image.SetOrigin((11.0, 23.0, -40.0))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.20261010.1"
    for z in range(6):
        if gap and z == 3:
            continue
        plane = image[:, :, z]
        tags = {
            "0008|0060": "CT",
            "0008|103e": "synthetic CT",
            "0020|000d": uid + ".1",
            "0020|000e": uid,
            "0020|0032": "\\".join(
                map(str, image.TransformIndexToPhysicalPoint((0, 0, z)))
            ),
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0020|0013": str(z + 1),
            "0028|1052": "-1024",
            "0028|1053": "2",
        }
        for key, value in tags.items():
            plane.SetMetaData(key, value)

        writer.SetFileName(str(folder / f"{5 - z}.dcm"))
        writer.Execute(plane)
    return image, values


def test_dicom_sorts_physical_positions_and_calibrates_once(tmp_path):
    image, values = write_series(tmp_path)
    series = dicom_series(str(tmp_path))
    assert len(series) == 1
    volume = read_dicom(series[0])
    np.testing.assert_array_equal(volume.array, values)
    np.testing.assert_allclose(volume.spacing, image.GetSpacing())
    np.testing.assert_allclose(volume.origin, image.GetOrigin())
    assert volume.hu_calibrated
    assert volume.metadata["series_uid"] == series[0]["uid"]


def test_dicom_missing_slice_is_rejected(tmp_path):
    write_series(tmp_path, gap=True)
    with pytest.raises(ValueError, match="missing slices"):
        read_dicom(dicom_series(str(tmp_path))[0])
