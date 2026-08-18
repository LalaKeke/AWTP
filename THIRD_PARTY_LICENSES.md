# Third-party software

This repository builds on the following open-source projects. Their licenses
and citation requirements continue to apply.

| Project | Use in this repository | License/source |
|---|---|---|
| HiP-AD | Base perception and planning framework | Apache-2.0; root `LICENSE` |
| Bench2Drive | Dataset integration and closed-loop evaluator | `bench2drive/LICENSE` |
| Bench2DriveZoo | Base train/validation split metadata | https://github.com/Thinklab-SJTU/Bench2DriveZoo |
| CARLA | Closed-loop simulator | https://github.com/carla-simulator/carla |
| MMDetection3D / MMDetection / MMCV | Detection framework | respective upstream repositories |
| SparseDrive, UniAD, VAD | Components acknowledged by upstream HiP-AD | respective upstream repositories |
| Theseus | Optional differentiable optimization | https://github.com/facebookresearch/theseus |

Before publishing a release, confirm that every newly added PNN source file is
owned by the releasing authors or distributed with the permission of all
contributors. This file is not a substitute for that authorship review.
