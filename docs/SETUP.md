# Physical Setup & Environment

Recorded 2026-09-15 from two setup photos and the preflight script output.
The photos themselves are gitignored — this file is the durable record.

---

## The rig

Dark matte work mat on a table. Both arms clamped to the near edge, facing in.

### Arms — SO-101 leader/follower pair

| | Side | End effector | Role |
|---|---|---|---|
| **Follower** | **Left** | Gripper jaws | **The only arm that can manipulate.** Does 100% of the picking and placing. |
| **Leader** | **Right** | Handle (no gripper) | Teleop input. Squeeze handle → follower gripper closes. Cannot touch objects. |

The leader sits off the mat next to the keyboard — it's a human input device,
not part of the workspace.

> **Consequence:** there is no "one arm per plate" option. A single arm must
> reach the brick pile *and* both plates.

### Cameras

| Camera | Mount | Suggested role |
|---|---|---|
| Overhead | Clamped articulated boom arm, looking down at the mat | Brick localization — gives pixel coords for the pick |
| Inspect | Small cam on a mini tripod behind the mat, facing forward | Close/oblique view for defect inspection |

Role assignment is **not yet locked** — see PLAN.md Phase 1.

### Workspace objects

- **Blue plate** and **coral plate** — the two sort destinations (one per class).
- **~14 LEGO bricks**, assorted colors, mix of 2x2 and 2x4.
- Bricks are bright saturated colors on a dark matte mat — **high contrast**,
  which makes classical segmentation viable for localization. See PLAN.md Phase 3.

### ⚠ Layout risk — reach envelope

In the first photo the coral plate sat far right on the mat while the follower
is clamped at the **left** edge. That is a long lateral reach for an SO-101.

**Action:** before anything else, jog the follower to each intended plate
position and the far corners of the brick zone. Stage both plates in a tight arc
on the follower's side. Do not spread the layout across the full mat width.
Confirming reach costs 10 minutes; discovering the limit mid-demo costs the run.

---

## Compute environment

Ubuntu host, user `lrd-demo`. Python env `intel_arc_env`.

### Verified by the preflight script — `PASS 19 · FAIL 0 · WARN 1`

```
[PASS] NPU device node      /dev/accel/accel0 present
[PASS] GPU render nodes     /dev/dri/renderD128
[PASS] Group membership     user in render + video groups
[PASS] physical-ai-studio   repo present at ~/physical-ai-studio
[PASS] Physical AI Studio   UI build — deps present
✓ ALL REQUIRED CHECKS PASSED — stack is ready
```

NPU and iGPU are both exposed and the user has permission to use them. No driver
work needed.

### ⚠ The one warning — fix this before Phase 4

```
[WARN] physicalai import (backend venv)
       runtime/native component not found under
       ~/physical-ai-studio/application/backend/
       hint: re-run 2_install_software.sh
```

Non-fatal for preflight, but `physicalai` is the exact module the VLA layer needs.
Fix it early, while there's still time to debug.

> Transcribed from a photo of the screen at an angle — paths and the exact
> wording may be slightly off. Re-run the preflight on the box to confirm.

### Sanity check to run first on the Ubuntu box

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

Expect `['CPU', 'GPU', 'NPU']`. If NPU is missing, stop and fix that before
building anything — 20 points depend on it.
