# Head-pose geometry dependencies and licenses

No third-party source file was copied into this repository.

| component | use | license/provenance |
|---|---|---|
| `thohemp/6DRepNet360` | architecture, 6D columns, SO(3), geodesic loss | MIT; audited commit `b06e4fed77f802fbf05319b85bb65215087d5c7a` |
| PINTO model-zoo `300_6DRepNet` | public converted full-rotation ONNX artifact | folder carries Thorsten Hempel MIT license and points to the official source; conversion is identified explicitly |
| `KupynOrest/head_detector` | optional VGGHeads architectural/second-teacher reference | source MIT; published VGGHeads weights are CC-BY-NC-4.0 and are not downloaded/enabled |
| PyTorch/torchvision | trainable models and tensor geometry | BSD-style |
| ONNX | strict converted-weight graph loading | Apache-2.0 |
| ONNX Runtime | offline equivalence/teacher execution | MIT |
| NumPy/SciPy/Pillow | annotation geometry, matching, images | BSD-style / HPND |

The official 6DRepNet360 checkpoint link was unavailable during implementation.
The workspace therefore contains terminal-downloaded, git-ignored converted
ONNX files, not a falsely relabeled official `.pth`. Their SHA-256 values are:

- fixed batch:
  `422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82`;
- dynamic batch:
  `852ea8e28014cee888a779fa235082fe3424e0ef2dd786c5c5496745741a46d8`.

The public conversion source is PINTO model-zoo
`300_6DRepNet`; its `download.sh` points to
`https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/300_6DRepNet/resources.tar.gz`.
That full archive is about 5.1 GB, so this repository does not provide an
automatic downloader that would silently transfer it. Users explicitly place
one required ONNX file under a local checkpoint path, then run
`scripts/verify_head_pose_weights.py`. Model files remain excluded by
`.gitignore`.
