# Demo Notes — measured results

Everything here is measured on the rig, not estimated. These are the numbers to
say out loud; the rubric asks for architecture, workload placement, optimization
choices, and observed results, and this is the last two.

---

## Architecture — one closed loop

```
PERCEIVE          DETECT               REASON            ACT              OPTIMIZE
wrist camera  ->  Anomalib PatchCore   score -> verdict  arm places the   OpenVINO IR,
at a fixed        on OpenVINO / NPU    -> instruction    brick in the     NPU + GPU + CPU
inspect pose      anomaly score        string            matching plate
```

The seam between detection and action is a **language instruction**, not a
function call:

| verdict | instruction handed to the policy | plate |
|---|---|---|
| defective | `"put the brick in the coral plate"` | coral |
| good | `"put the brick in the blue plate"` | blue |
| unsure | re-inspect, then fall back to `defective` | — |

Same policy, same scene, different destination — selected purely by text. That
is how Anomalib's output reaches the robot.

---

## Defect detection — Anomalib PatchCore

Trained on **normal samples only**. No defective examples are used for fitting;
they exist solely to place the threshold.

| | |
|---|---|
| Model | PatchCore, `wide_resnet50_2`, layers 2+3 |
| Training set | 40 normal crops — 5 bricks, 2 per brick held out |
| Held out | 10 normal crops, never fitted |
| Validation | 50 defective crops, threshold tuning only |

Measured score distributions:

| class | n | min | mean | max |
|---|---|---|---|---|
| good (fitted) | 40 | 0.0000 | 0.0627 | 0.4883 |
| **good (held out)** | 10 | 0.0000 | 0.2228 | 0.3960 |
| defective | 50 | 0.4324 | 0.8036 | 1.0000 |

Threshold `0.4142`, band `±0.012`. At those settings **every held-out crop and
every defect classifies correctly** — 10/10 and 50/50.

### Why the held-out row is the one that matters

An earlier version of this dataset was ~47 crops of a *single* brick, with
nothing held out, and it reported AUROC 1.00 with good scoring 0.0000–0.0000.
That number was meaningless: PatchCore scores by distance to its memory bank,
and every test image's own patches were *in* that bank. It was measuring
memorization. The failure showed up physically — a different good brick was
classified defective.

So the set is now 5 distinct bricks with 2 crops per brick withheld from the
fit. The held-out row is the only one that predicts live behaviour.

Three fitted normals score above threshold. Inspecting them is instructive: the
worst (0.4883) is a 2×4 brick sitting half out of frame while the rest of the
set is 2×2 and centred. A high anomaly score for a different geometry at a
different position is the model working, not failing. The band is kept narrow
because widening it enough to absorb those three would swallow the lowest
defects at 0.4324.

Why PatchCore rather than PaDiM: the defect is marker marks on the stud faces —
localized structural difference, which nearest-neighbour patch matching
localizes well.

---

## Latency and device placement

30 inferences per device, after warm-up so NPU graph compilation is excluded.

| Stage | Device | Mean | p50 | p95 | Throughput |
|---|---|---|---|---|---|
| PatchCore, 90-normal bank | **Arc iGPU** | 174 ms | - | - | 5.7 /s |
| PatchCore, 47-normal bank | NPU | 51.18 ms | 51.20 ms | 51.50 ms | 19.5 /s |
| PatchCore | GPU (Arc) | 5.41 ms | 4.85 ms | 6.82 ms | 184.9 /s |
| PatchCore | CPU | 84.20 ms | 83.86 ms | 89.87 ms | 11.9 /s |

### SmolVLA policy, Arc iGPU (torch-xpu)

60 control steps after warm-up, chunk size `n_action_steps=50`:

| | n | mean |
|---|---|---|
| queue drain (no VLM) | 58 | **2.24 ms** |
| VLM forward | 2 | **176.0 ms** |
| overall | 60 | 8.0 ms |

The 30 Hz control budget is 33.3 ms per step. SmolVLA predicts a *chunk* of 50
actions, so the ~500 M parameter VLM runs once per 1.7 s of motion and the other
49 steps are a queue pop costing ~2 ms. Mean cost is a quarter of budget; the
visible cost is one ~143 ms stall per chunk boundary, not a sustained overrun.

This is the argument for keeping the iGPU free for the policy: it is the only
stage with a real throughput requirement.

### Why the detector moved off the NPU

This is the interesting part of the placement story, and it is a deliberate
choice rather than an oversight:

- **The detector runs once per brick, not per frame.** 51 ms of a roughly 15 s
  pick-and-place cycle is 0.3% of the loop. Making it 5 ms saves nothing that
  anyone can observe.
- **The iGPU is the scarce resource.** The VLA policy is a ~500 M parameter
  vision-language model that needs sustained throughput at control rate. Putting
  a once-per-brick detector on the same engine would contend with it.
- **The CPU stays free for control.** Camera decode, the 30 Hz action loop and
  the serial bus are latency-sensitive and irregular — exactly the work a CPU
  should keep.

So all three engines do the work each is suited to. The GPU number is reported
here because measuring it is what makes the choice defensible: PatchCore's
nearest-neighbour search maps poorly to the NPU's fixed-function units, and that
is visible in the 10× gap. It simply does not matter at this duty cycle.

---

## Reliability

- **Median vote, not a single frame** — every verdict is the median of 12
  consecutive scores (`detector.vote_frames`). One frame is a poor basis for a
  decision: glare on a stud or an autoexposure step moves the score enough to
  flip a borderline brick. Median discards those outliers where a mean would let
  one bad frame drag the decision across the line. ~600 ms once per brick.
- **Uncertainty band** — scores within ±0.012 of the threshold are never guessed.
  The loop re-inspects, and only then falls back to the configured class.
- **Pose repeatability** — the wrist camera moves with the arm, so the loop
  returns to a fixed inspect pose before every score. Without it the detector
  compares a close-up against a wide shot and calls the difference an anomaly.
- **Safe motion** — all point-to-point moves interpolate rather than step. A step
  command to a distant target pulls maximum current at once, which tripped
  `shoulder_lift`'s overload protection during development and dropped it off
  the servo bus.

---

## Physical AI Studio and OpenVINO — what runs where, precisely

Worth stating exactly, because a vague claim here is worse than an accurate
split.

| Component | Reality |
|---|---|
| Dataset | LeRobot v3, the format Studio manages. Readable by Studio and by `physicalai.data.lerobot.LeRobotDataModule` with no import step |
| Studio backend + UI | Running, project and both arms registered |
| Policy training | `lerobot-train`. The Studio path (`scripts/16_studio_train.sh`) is built and reaches the Lightning trainer, but was not the run used for the demo |
| Detector | **OpenVINO IR on the NPU** — `experiments/detector/openvino_fp16/weights/openvino/model.xml` |
| Policy | **PyTorch on the Arc iGPU** (torch-xpu), not OpenVINO |

### Why the policy is not OpenVINO IR

Not for lack of trying, and the blocker is specific. `physicalai export` loads
through Lightning's `load_from_checkpoint`, so it only accepts a `.ckpt` from
`physicalai fit`; our policy came from `lerobot-train`, which writes a LeRobot
`pretrained_model/` directory.

`scripts/21_export_policy_ov.py` works around that by calling the Policy mixin's
`export()` on an instance, which needs only a loadable checkpoint. It gets as
far as loading the trained weights, then stops in
`_get_default_export_input_sample`: the policy cannot synthesise a sample input
without dataset features, so tracing needs an explicit one.

A related incompatibility is documented in `scripts/20_prepare_vla_base.py`:
`physicalai` cannot read `lerobot/smolvla_base`'s config.json at all, because
`SmolVLAConfig.from_dict(..., strict=False)` does not drop the LeRobot-only keys
its comment claims it drops — it forwards them to jsonargparse, which rejects
the config. That one we did solve, by filtering the config to the fields
`SmolVLAConfig` declares.

So: three engines, each doing work it suits, with measured numbers for all of
them — and one honest gap.

## Reproducing these numbers

```bash
python scripts/11_benchmark.py --iters 30
```

Runs the device sweep and re-scores every captured crop, reporting separation
and a suggested threshold.
