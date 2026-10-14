# Third-party resources

The Apache-2.0 license in this repository covers openKaloLumen source code.
It does not relicense dependencies, sample images, or model weights. The source
and wheel distributions exclude downloaded models, datasets, and reference
repositories. Dependency notices remain in the dependencies' installed metadata;
the desktop bundle also includes those metadata files and the Qt license texts
in `packaging/licenses/`.

| Component | License / terms | Source |
| --- | --- | --- |
| NumPy | BSD-3-Clause | https://numpy.org/doc/stable/license.html |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| VTK | BSD-3-Clause | https://vtk.org/about/ |
| SimpleITK | Apache-2.0 | https://github.com/SimpleITK/SimpleITK/blob/master/LICENSE |
| Pillow | MIT-CMU | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| PySide6-Essentials / Qt for Python | LGPL-3.0 / GPL-3.0 / commercial, depending on component | https://doc.qt.io/qtforpython-6/licenses.html |
| Optional PyTorch | BSD-3-Clause and bundled component notices | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Optional MONAI | Apache-2.0 | https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE |
| Optional NiBabel | MIT | https://github.com/nipy/nibabel/blob/master/COPYING |
| Optional einops | MIT | https://github.com/arogozhnikov/einops/blob/main/LICENSE |
| Optional safetensors | Apache-2.0 | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| PyInstaller (build tool) | GPL-2.0 with bootloader exception | https://pyinstaller.org/en/stable/license.html |

Qt libraries are bundled as shared libraries by PyInstaller. The project uses
Qt Widgets and excludes Qt WebEngine, Qt QML, and Qt Quick from its desktop bundle.
The Qt LGPL/GPL texts, a Qt attribution notice, and upstream source locations
are in `packaging/licenses/`. Preserve these license texts and notices when distributing binaries.
The exact dependency licenses and bundled components must be checked for each
release; this table is a source index, not a replacement for upstream notices.

## Sample images

Samples are downloaded on request and validated by SHA-256. Their source URLs
and checksums are in `src/openkalolumen/samples.py`, and a `.source.json` sidecar
records provenance. Follow the data source's terms for use and redistribution:

- [3D Slicer SampleData definitions](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py)
- [SlicerTestingData](https://github.com/Slicer/SlicerTestingData)
- [SlicerHeart PDA sample definitions](https://github.com/SlicerHeart/SlicerHeart/blob/master/ValveAnnulusAnalysis/ValveAnnulusAnalysis.py)

## Optional segmentation model

The integration calls MONAI's VISTA3D network with separately obtained
NV-Segment-CTMR weights. The upstream README distinguishes Apache-2.0 **code**
from **weights** under the NVIDIA OneWay Non-Commercial License for academic
research. Download the model yourself after reviewing the upstream terms;
weights and upstream repositories are not included in releases.

- [NVIDIA NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
- [NVIDIA model license](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf)

openKaloLumen is an independent implementation and is not affiliated with
KaloLumen, NVIDIA, MONAI, Qt, or 3D Slicer.
