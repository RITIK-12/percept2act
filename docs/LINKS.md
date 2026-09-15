# Reference Links

Curated documentation for the Intel Physical AI Challenge build.
All Intel links verified `200 OK` on 2026-09-15.

> **Version note:** Intel Open Edge Platform docs are served per-version. These
> point at `/dev/`. Swap `/dev/` for a release tag if `dev` drifts.

---

## Start here

| Link | What it covers |
|---|---|
| [Robotics AI Suite — landing](https://docs.openedgeplatform.intel.com/dev/ai-suite-robotics.html) | Top-level index for the whole suite |
| [System Requirements](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/platform_foundation/system_requirements.html) | Supported CPUs/OS before installing anything |
| [Getting Started](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/platform_foundation/getting_started.html) | Install paths overview |
| ├ [Express install](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/platform_foundation/getting_started/express.html) | Fastest path — use this on Challenge Day |
| ├ [Step-by-step install](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/platform_foundation/getting_started/step_by_step.html) | When express fails |
| └ [Image Composer Tool](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/platform_foundation/getting_started/image_composer_tool.html) | Building a bootable image |

---

## 1. PERCEIVE — cameras

| Link | What it covers |
|---|---|
| [Sensors (cameras, LiDAR)](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/components/sensors/index.html) | Camera bring-up and drivers |
| [Segmentation + RealSense tutorial](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/reference_applications/segmentation_realsense_tutorial.html) | RealSense → OpenVINO pipeline |
| [Multi-camera OpenVINO demo](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/reference_applications/openvino_multicam_demo.html) | **Two-camera setup — matches our rig** |

## 2. DETECT — defect / anomaly detection

| Link | What it covers |
|---|---|
| [Anomalib (GitHub)](https://github.com/open-edge-platform/anomalib) | Source, model zoo, examples |
| [Anomalib docs](https://anomalib.readthedocs.io/) | Training + export API |
| [Object detection tutorial](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/reference_applications/object_detection_tutorial.html) | Locating blocks on the mat |
| [YOLOv8 + OpenVINO tutorial](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/reference_applications/yolov8_openvino_tutorial.html) | Detector → OpenVINO conversion |

> Anomalib is **not** linked from the Intel robotics docs tree — it lives in its
> own repo under the same `open-edge-platform` org.

## 3. REASON — VLA / Physical AI Studio

| Link | What it covers |
|---|---|
| [Physical AI Studio (GitHub)](https://github.com/open-edge-platform/physical-ai-studio) | **Primary repo — the VLA workflow tool** |
| [Physical AI (docs section)](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/physical-ai/index.html) | Landing page for the Physical AI track |
| [Agentic Skills](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/skills/index.html) | Skill abstraction feeding robot actions |

## 4. ACT — arm control

| Link | What it covers |
|---|---|
| [Stationary Arm blueprint](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/hardware_blueprints/stationary_arm/index.html) | Hardware blueprint for arm rigs |
| [UR5e + Robotiq + RealSense](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/hardware_blueprints/stationary_arm/ur5e-robotiq-realsense.html) | Closest worked example (different arm) |
| [Stationary arm software refs](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/software_references/stationary_arm/index.html) | RVC sim + deploy |
| ├ [RVC simulation](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/software_references/stationary_arm/simulation/rvc_sim.html) | Dry-run without hardware |
| └ [RVC deployment](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/software_references/stationary_arm/deployment/rvc_deploy.html) | Deploy to real arm |
| [ROS 2 / Gazebo middleware](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/components/middleware/index.html) | Middleware layer |
| [LeRobot (GitHub)](https://github.com/huggingface/lerobot) | **SO-101 driver, calibration, teleop, ACT/DP training** |

> SO-101 is **not** documented in the Intel tree. LeRobot is the de-facto
> driver + training stack for SO-100/SO-101.

## 5. OPTIMIZE — OpenVINO on Core Ultra

| Link | What it covers |
|---|---|
| [OpenVINO in Robotics AI Suite](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/index.html) | Suite-specific OpenVINO usage |
| [OpenVINO developer tools](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/developer_tools/openvino.html) | Toolchain setup |
| [OpenVINO official docs](https://docs.openvino.ai/) | Conversion, NPU/GPU plugins, quantization |
| [Reference applications index](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/reference_applications/index.html) | All worked OpenVINO examples |
| [Benchmarking & System Profiler](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/components/benchmarking/index.html) | **Latency numbers for the demo writeup** |

---

## Pre-optimized robot policy models

Already OpenVINO-converted by Intel. Useful if we go the learned-policy route.

| Model | Link | Notes |
|---|---|---|
| Model index | [models/index](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/index.html) | Full list |
| **ACT** | [model_act](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_act.html) | **Best fit — standard SO-101 policy via LeRobot** |
| **Diffusion Policy** | [model_dp](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_dp.html) | Alternative to ACT |
| GraspNet | [model_graspnet](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_graspnet.html) | Grasp pose generation |
| Pi0 | [model_pi0](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_pi0.html) | Large VLA |
| Pi0.5 + RTC | [pi05_with_rtc](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/software_references/humanoid/sample_pipelines/pi05_with_rtc.html) | Real-time chunking |
| Pi0.5 optimization | [pi05-optimization](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/pi05-optimization.html) | Optimization writeup |
| GR00T-N1.7 | [model_gr00t_n1d7](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_gr00t_n1d7.html) | NVIDIA VLA, OV-converted |
| RDT-1B | [model_rdt](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_rdt.html) | Diffusion transformer |
| BC-RNN | [model_bc_rnn](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_bc_rnn.html) | Behavior cloning baseline |
| Depth Anything v2 | [model_depthanythingv2](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_depthanythingv2.html) | Monocular depth |

---

## Gaps

Not covered anywhere in the Intel docs tree — we supply these ourselves:

- **SO-101** bring-up, calibration, teleop → LeRobot
- **Anomalib ↔ Physical AI Studio** wiring → no worked example exists
