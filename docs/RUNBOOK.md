# Challenge Day Runbook

Exact sequence. Every command is copy-paste runnable. Do them in order — each
step's gate is the next step's precondition.

**Architecture:** one brick at a time, camera → Anomalib → verdict → instruction
string → SmolVLA → correct plate.

```
PERCEIVE      DETECT              REASON            ACT              OPTIMIZE
front cam --> PatchCore/NPU  -->  verdict picks --> SmolVLA places -> OpenVINO,
              anomaly score       an instruction    the brick          NPU+GPU+CPU
```

The Anomalib → VLA seam is the **instruction string**. A defective verdict
selects `"put the brick in the coral plate"`; a good one selects
`"put the brick in the blue plate"`. Same policy, same scene, different
destination — chosen by language. That is the thing to point at when a judge
asks how defect output reaches the robot.

Everything runs in the **`hack_lerobot`** conda env, which has lerobot, Anomalib
and OpenVINO together. Set it up once per terminal:

```bash
conda activate hack_lerobot && cd ~/percept2act
```

---

## Step 0 · Preflight — 2 min

```bash
cd ~/percept2act && bash scripts/00_preflight.sh
```

Expect `FAIL 0`. It checks NPU visibility, both arm ports, existing calibration,
all three cameras, and re-applies the wrist camera's exposure (its v4l2 settings
reset on every replug).

One durable fix worth doing now — you are not in `dialout`, so the arm ports
work only because they are currently `chmod 777`, which udev undoes on replug:

```bash
sudo usermod -aG dialout ird-demo
```

Log out and back in for it to take effect.

**Gate:** `FAIL 0`.

---

## Step 1 · Confirm teleop still works — 3 min

Both arms were calibrated as `hack_follower` / `hack_leader`, so this should
just work. Squeeze the leader's handle and watch the follower's gripper close.

```bash
conda activate hack_lerobot && lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=hack_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM0 --teleop.id=hack_leader
```

Note the ports: **follower is ttyACM1, leader is ttyACM0**. The config uses
stable `/dev/serial/by-id/` paths so a replug cannot silently swap them.

**Gate:** leader drives follower; gripper opens and closes.

---

## Step 2 · Start Physical AI Studio — 5 min

Two terminals. Recording through the Studio rather than the bare CLI is what the
20-point Studio criterion rewards, and you get live camera feeds while you
teleoperate.

```bash
bash scripts/04_start_studio.sh backend
```

```bash
bash scripts/04_start_studio.sh ui
```

Then open <http://localhost:5173>. Check both are up with:

```bash
bash scripts/04_start_studio.sh check
```

If the UI fights you, fall back to the CLI in Step 3 — it writes the same
LeRobot-format dataset and costs you nothing except the Studio talking point.

**Gate:** UI loads and sees the follower.

---

## Step 3 · Record the sorting dataset — 60–75 min ⚠ critical path

This needs a human on the leader arm, so start it as early as possible. One
dataset, **two instruction strings**, ~30 episodes each.

Per episode: one brick on the mat → pick it up → place it in the named plate →
return toward home. Vary brick colour, size and starting position. Keep the
visual scene as similar as you can between the two classes, so the policy learns
to read the *instruction* rather than the scene.

```bash
bash scripts/03_record_stage2.sh defective 30
```

```bash
bash scripts/03_record_stage2.sh good 30
```

The second run appends to the same dataset via `--resume`. Verify the balance
before you spend an hour training on it:

```bash
bash scripts/03b_verify_stage2.sh
```

A one-sided dataset trains happily and then **ignores the instruction** at
inference, which looks exactly like "the VLA doesn't work" during the demo.

This dataset is **defect-independent** — it stays valid whatever the defect
definition is, because the instruction names the plate.

**Gate:** ~60 episodes, both task strings present, roughly balanced.

---

## Step 4 · Train SmolVLA — 45–120 min, unattended

Fine-tunes from `lerobot/smolvla_base` (already in the HF cache). Own terminal;
move straight on to Step 5 while it runs.

```bash
bash scripts/05_train.sh smolvla --dry-run
```

```bash
bash scripts/05_train.sh smolvla
```

Runs on the Arc iGPU via torch-xpu. If a flag is rejected, check exact names
with `lerobot-train --help` — the script prints the full command first.

**Gate:** checkpoints appearing under `experiments/smolvla_sort/checkpoints/`.

---

## Step 5 · Capture bricks for Anomalib — 20 min

Do this **while SmolVLA trains**. First set the inspection crop — the region of
the front camera where a brick sits when being inspected:

```bash
python scripts/01_assign_cameras.py --tune-crop 180,120,460,360
```

Adjust the numbers until `experiments/camera_preview/inspect_crop.png` tightly
frames a brick with a little margin, then write them into
`inspection_station.crop` in [`config/scenario.yaml`](../config/scenario.yaml).

Then capture. PatchCore fits on **normal samples only** — slide a good brick
around the crop region while the auto-capture runs:

```bash
python scripts/06_capture_normals.py --class good --auto 60 --interval 1.0
```

Also grab a handful of damaged bricks. These are **not** used for fitting, only
for choosing the threshold — which is the difference between a detector that
works and one that guesses:

```bash
python scripts/06_capture_normals.py --class defective --auto 15
```

**Gate:** ~50 good crops, ~10 defective crops, all tightly framed.

---

## Step 6 · Fit PatchCore and export to OpenVINO — 15 min

```bash
python scripts/07_train_anomalib.py
```

No backprop — it builds a memory bank of patch features, so this takes minutes.
PatchCore rather than PaDiM because the defect is physical damage: localized
structural differences, which nearest-neighbour patch matching localizes well.

It prints the threshold it computed. **Put that number into
`detector.threshold`** in `config/scenario.yaml`.

**Gate:** OpenVINO IR under `experiments/detector/`, threshold recorded.

---

## Step 7 · Validate the detector and pick the device — 10 min

```bash
python scripts/09_benchmark.py
```

Two outputs:

- **latency on NPU vs GPU vs CPU** — the table you must explain live
- **score separation** between good and damaged bricks, with a suggested
  threshold and uncertainty band

If the two distributions overlap, **widen `detector.uncertainty_band`** so
borderline scores route to `unsure` instead of being guessed. An honest unsure
path scores under the reliability criterion; a confident wrong answer does not.

**Gate:** good and defective scores separate, threshold and band written into config.

---

## Step 8 · Close the loop — 30 min

Three stages, so a failure is always localized.

**8a — detector and reasoning only, arm untouched:**

```bash
python scripts/10_run_loop.py --dry-run --bricks 5
```

Watch each brick produce a score, a verdict, and the instruction it selects.

**8b — one brick, robot live.** Keep a hand near the follower's power:

```bash
python scripts/10_run_loop.py --bricks 1
```

**8c — the full run:**

```bash
python scripts/10_run_loop.py --bricks 14
```

Prints a per-brick report and the stage/device/latency table.

**Gate:** bricks land in the correct plates without a human touching the arm.

---

## Step 9 · Demo prep — 30 min

Run 8c three times and **record the success rate honestly** — a known 11/14 with
an explanation beats a vague claim of "it works".

Rehearse the four things the rubric asks for, in this order:

1. **Architecture** — the five stages above, one integrated loop
2. **Workload placement** — PatchCore on NPU (fixed shape, sustained, frees the
   CPU), SmolVLA on the Arc iGPU (~500M VLM, needs throughput), control loop and
   camera decode on CPU (latency-sensitive, irregular)
3. **Optimization choices** — OpenVINO IR export, FP16, device sweep in Step 7,
   INT8 attempted via `--int8`
4. **Observed results** — the latency table and the success rate

Then write down the reset procedure between runs so a retry is calm.

---

## Fallback ladder

Cut from the bottom. Never cut the loop.

| # | Capability | If you have to drop it |
|---|---|---|
| 1 | Closed loop, one brick, perception → action | **Non-negotiable.** Without this there is no submission. |
| 2 | Anomalib on NPU via OpenVINO with recorded latency | Run the detector on CPU; keep the loop. |
| 3 | SmolVLA doing the instruction-conditioned sort | This is the 20-point bucket — keep it over everything below. |
| 4 | Uncertainty band and re-inspection | Cheap to keep; most teams skip it, so it differentiates. |
| 5 | INT8 quantization | Drop freely. |
| 6 | Multi-brick continuous sorting | Demo one brick at a time. |
| 7 | ACT present-step (`scripts/02_record_stage1.sh`) | Optional upgrade; improves detector accuracy by fixing brick pose. |

If only one learned policy fits in the day, keep **SmolVLA** and script the
present step. SmolVLA is where the VLA points are; the present motion is the one
a script does adequately.

---

## When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| Arm port permission denied | udev reset the `chmod`; you are not in `dialout` | `sudo usermod -aG dialout ird-demo`, re-login |
| Wrist camera all black | lens cap, or exposure reset on replug | Remove cap; `bash scripts/00_preflight.sh` re-applies exposure |
| Wrist camera blurry | lens not focused | Twist the barrel; it is excluded from `policy_inputs` by default |
| `device 'NPU' not available` | OpenVINO cannot see the NPU | `python -c "import openvino as ov; print(ov.Core().available_devices)"` |
| Detector loads but every score is identical | crop is empty or off-target | `python scripts/01_assign_cameras.py --tune-crop ...` |
| Policy ignores the instruction | dataset one-sided | `bash scripts/03b_verify_stage2.sh`, record the missing class |
| Arm starts a pick mid-place | action chunk buffer carried over | Already handled by `PolicyRunner.reset()` between bricks |
| `import physicalai` fails | venv was copied from another machine | `~/hackathon_install/2_install_software.sh 5` |

---

## Verified rig facts

Measured on the box, not assumed.

| Thing | Value |
|---|---|
| Follower arm | `/dev/ttyACM1`, id `hack_follower`, calibrated |
| Leader arm | `/dev/ttyACM0`, id `hack_leader`, calibrated |
| Front camera (tripod, sees workspace) | RealSense serial `243722062690` |
| Boom camera (overhead) | RealSense serial `243622062278` — **aim it at the mat** |
| Wrist camera | Sonix UVC, `/dev/video12`, exposure 60, **needs focusing** |
| OpenVINO devices | `['CPU', 'GPU', 'NPU']` |
| Runtime env | `hack_lerobot` — lerobot 0.6.1, anomalib 2.6.2, openvino 2026.3.1 |
| Studio env | `~/physical-ai-studio/application/backend/.venv` — physicalai + lerobot |
| RealSense config discriminator | `intelrealsense`, **not** `realsense` |
