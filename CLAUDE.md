# percept2act

Intel Physical AI Challenge entry: an SO-101 arm that inspects LEGO bricks,
classifies them defective / good, and sorts each into the matching plate.

**Perceive → Detect → Reason → Act → Optimize.**

## Read these first

| File | Why |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | The build plan. Phases, time budget, fallback ladder. |
| [docs/SETUP.md](docs/SETUP.md) | The physical rig + verified environment state. |
| [docs/CHALLENGE.md](docs/CHALLENGE.md) | Requirements + judging rubric. |
| [docs/LINKS.md](docs/LINKS.md) | Verified Intel / OpenVINO / LeRobot / Anomalib docs. |

## Hardware facts that constrain everything

- **Left arm = FOLLOWER.** Has the gripper. This is the only arm that can touch a brick.
- **Right arm = LEADER.** Handle, no gripper. Teleop input device only.
- **One arm does all the sorting** — it must reach the brick pile *and* both plates.
- Two cameras: an overhead boom cam and a tripod cam facing the workspace.
- Two plates: blue and coral. One per class.
- Compute: Intel Core Ultra Series 3. NPU at `/dev/accel/accel0`, Arc iGPU at
  `/dev/dri/renderD128`. Both verified present.

## The strategy in one paragraph

65 of the 100 points are **integration**, not model accuracy. The exact defect
definition is withheld until Challenge Day kickoff. So: build the full closed
loop end-to-end with a stub detector *first*, keep every scenario-specific value
in [config/scenario.yaml](config/scenario.yaml), and swap in the real Anomalib
model once the defect is revealed. A working loop with a mediocre detector
scores far better than a great detector with no loop.

## Conventions

- Nothing scenario-specific gets hardcoded. It goes in `config/scenario.yaml`.
- Calibration values live in `config/calibration.yaml`, never in source.
- Every inference stage logs its device (NPU/GPU/CPU) and latency — the rubric
  requires explaining workload placement during the live demo.

## Environment

Ubuntu box, user `lrd-demo`. Physical AI Studio at `~/physical-ai-studio`.
Python env `intel_arc_env`.

Known issue, see docs/SETUP.md: `physicalai` fails to import in the backend venv.
Fix before starting Phase 4.
