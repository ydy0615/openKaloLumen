from pathlib import Path

from openkalolumen.config import AppPaths


def test_explicit_paths_override_environment_without_creating_directories(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENKALOLUMEN_HOME", str(tmp_path / "env-data"))
    monkeypatch.setenv("OPENKALOLUMEN_MODEL", str(tmp_path / "env-model"))
    monkeypatch.setenv(
        "OPENKALOLUMEN_SEGMENTATION_PYTHON", str(tmp_path / "env-python")
    )
    paths = AppPaths.resolve(tmp_path / "data", tmp_path / "model", tmp_path / "python")
    assert paths == AppPaths(tmp_path / "data", tmp_path / "model", tmp_path / "python")
    assert not paths.data.exists()


def test_environment_paths_allow_installed_app_to_use_external_inference(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENKALOLUMEN_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENKALOLUMEN_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("OPENKALOLUMEN_SEGMENTATION_PYTHON", str(tmp_path / "python"))
    assert AppPaths.resolve() == AppPaths(
        tmp_path / "data", tmp_path / "model", tmp_path / "python"
    )


def test_defaults_do_not_depend_on_checkout_or_working_directory(tmp_path, monkeypatch):
    for variable in (
        "OPENKALOLUMEN_HOME",
        "OPENKALOLUMEN_MODEL",
        "OPENKALOLUMEN_SEGMENTATION_PYTHON",
    ):
        monkeypatch.delenv(variable, raising=False)
    expected = AppPaths.resolve()
    monkeypatch.chdir(tmp_path)
    assert AppPaths.resolve() == expected
    assert expected.data.is_absolute()
    assert expected.model == expected.data / "models/NV-Segment-CTMR"
    assert Path(expected.segmentation_python).is_file()
