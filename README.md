# percept2act

**Perceive → Detect → Reason → Act → Optimize.**

An SO-101 arm inspects a LEGO brick, classifies it defective or good with
Anomalib, and places it in the matching plate — where the link between
perception and manipulation is an **English instruction string**, not a function
call. Inference is OpenVINO-optimized and distributed across the NPU, Arc iGPU
and CPU of an Intel Core Ultra Series 3.

Built for the Intel Physical AI Challenge.

---

## The closed loop

```
PERCEIVE          DETECT               REASON              ACT              OPTIMIZE
wrist camera  ->  Anomalib PatchCore   score -> verdict    SmolVLA places   OpenVINO IR,
at a fixed        on OpenVINO / NPU    -> instruction      the brick in     NPU + GPU + CPU,
inspect pose      median of 12 frames  string              the named plate  per-stage logging
```

The seam between detection and action is a sentence:

| verdict | instruction handed to the policy | plate |
|---|---|---|
| defective | `"put the brick in the coral plate"` | coral |
| good | `"put the brick in the blue plate"` | blue |
| unsure | re-inspect, then fall back to `defective` | — |

Same checkpoint, same scene, different destination — selected purely by text.
Defined in [config/scenario.yaml](config/scenario.yaml) (`task.instructions`),
resolved at [config.py:65](src/percept2act/config.py:65), handed to the policy at
[orchestrator.py:258](src/percept2act/orchestrator.py:258).

---

## Quickstart

```bash
conda activate hack_lerobot && cd ~/percept2act
```

```bash
python scripts/22_demo_wall.py --executor policy
```

Place a brick at the inspect spot. It settles, PatchCore scores it 12 times on
the NPU, the median verdict picks an instruction, and SmolVLA places the brick.
`q` stops. Full command reference in [docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Requirements → where they are met

The challenge brief lists four deliverables and a mandated stack. This section
maps each to the code that implements it.

### 1. Working closed-loop prototype, perception → action

One process, one command, no manual step between seeing and acting.

| Piece | File |
|---|---|
| The loop itself | [src/percept2act/orchestrator.py](src/percept2act/orchestrator.py) |
| Brick-presence gate (HSV value, colour-agnostic) | [cameras.py `brick_fraction`](src/percept2act/cameras.py) |
| Live demo entry point | [scripts/22_demo_wall.py](scripts/22_demo_wall.py) |
| Minimal demo entry point | [scripts/12_demo.py](scripts/12_demo.py) |

`Orchestrator.run()` sequences `goto_inspect_pose → resolve_verdict → place` per
brick. The arm returns to a fixed pose before every inspection, because the
wrist camera moves with the arm and scores are only comparable from one pose
([orchestrator.py:170](src/percept2act/orchestrator.py:170)).

### 2. VLA + Anomalib integration

| Piece | File |
|---|---|
| Anomalib PatchCore fit + OpenVINO export | [scripts/10_train_detector.py](scripts/10_train_detector.py) |
| Detector inference wrapper | [src/percept2act/detector.py](src/percept2act/detector.py) |
| Verdict → instruction → policy | [orchestrator.py:256](src/percept2act/orchestrator.py:256) |
| SmolVLA inference | [src/percept2act/policy_runner.py](src/percept2act/policy_runner.py) |
| Swappable ACT backends (`stub` / `replay` / `policy`) | [src/percept2act/executor.py](src/percept2act/executor.py) |

The policy runs through LeRobot's own processor pipeline, so tokenization and
normalization match training exactly
([policy_runner.py `_make_processors`](src/percept2act/policy_runner.py)).

### 3. OpenVINO-optimized deployment on Core Ultra 3

| Piece | File |
|---|---|
| `ov.Core()`, device validation, `compile_model` | [detector.py:82–100](src/percept2act/detector.py:82) |
| Static reshape for the NPU plugin | [detector.py `_compile`](src/percept2act/detector.py) |
| Device placement policy | [config/scenario.yaml](config/scenario.yaml) `openvino.devices` |
| Three-device benchmark | [scripts/11_benchmark.py](scripts/11_benchmark.py) |
| Per-stage device + latency logging | [src/percept2act/latency.py](src/percept2act/latency.py) |

Every stage records the device it ran on to `experiments/latency.jsonl`
([orchestrator.py:179, :201, :260](src/percept2act/orchestrator.py:201)):

```
detect.patchcore_vote   NPU
act.policy              GPU
act.goto_inspect        CPU
```

### 4. Live demo + architecture/optimization explanation

| Piece | File |
|---|---|
| Measured results, device placement rationale | [docs/DEMO.md](docs/DEMO.md) |
| Three-camera demo view | [scripts/22_demo_wall.py](scripts/22_demo_wall.py) |
| Requirements and rubric reference | [docs/CHALLENGE.md](docs/CHALLENGE.md) |

The demo window shows the scored crop, the live score against the threshold, the
device and its latency, and the instruction string driving the arm — so the
pipeline is legible while it runs, not just afterwards.

---

## Mandated stack

| Layer | Component | Where it is used |
|---|---|---|
| Arm | SO-101 leader/follower, LeRobot | [arms in config](config/scenario.yaml), [19_record_clean.py](scripts/19_record_clean.py) |
| Anomaly detection | **Anomalib** PatchCore, `wide_resnet50_2` | [10_train_detector.py](scripts/10_train_detector.py) |
| Optimization | **OpenVINO** IR, FP16 | [detector.py](src/percept2act/detector.py), [11_benchmark.py](scripts/11_benchmark.py) |
| Orchestration / VLA | **Intel Physical AI Studio** (`physicalai`) | [13](scripts/13_start_studio.sh), [15](scripts/15_studio_record.sh), [16](scripts/16_studio_train.sh), [17](scripts/17_export_policy.sh), [20](scripts/20_prepare_vla_base.py), [21](scripts/21_export_policy_ov.py) |
| Silicon | Core Ultra Series 3 — NPU + Arc iGPU + CPU | `openvino.devices` in [config](config/scenario.yaml) |

---

## Measured results

Full numbers, method and rationale in **[docs/DEMO.md](docs/DEMO.md)**.

**Defect detection** — PatchCore fits on normal samples only; defective crops
are used solely to place the threshold. Validated on bricks withheld from the
fit, because scoring the fitted set measures memorization, not generalization.

**Latency and placement** — 30 inferences per device, post-warmup:

| Stage | Device | Mean | Why there |
|---|---|---|---|
| PatchCore | **NPU** | 51.18 ms | once per brick; frees the iGPU for the policy |
| PatchCore | GPU (Arc) | 5.41 ms | measured for comparison, not used |
| PatchCore | CPU | 84.20 ms | measured for comparison, not used |
| SmolVLA | **Arc iGPU** | 2.24 ms / 176 ms | 176 ms when the VLM runs, 2.24 ms for the other 49 steps of each action chunk |
| Control loop, camera decode | **CPU** | — | latency-sensitive and irregular |

The detector is deliberately *not* on the faster GPU: it runs once per ~15 s
pick-and-place cycle, while the ~450 M parameter policy needs sustained iGPU
throughput at control rate.

**Reliability** — verdicts are the median of 12 frames, scores inside the
uncertainty band are re-inspected rather than guessed, and all point-to-point
moves interpolate rather than step (a step command to a distant target trips
`shoulder_lift`'s overload latch).

---

## Repo layout

```
config/scenario.yaml          every scenario-specific value; no constants in source
src/percept2act/
  orchestrator.py             the closed loop
  detector.py                 Anomalib IR via OpenVINO, device-pinned
  executor.py                 stub | replay | policy, one interface
  policy_runner.py            SmolVLA through LeRobot's processor pipeline
  cameras.py                  camera abstraction + brick occupancy metric
  motion.py                   interpolated moves, episode playback
  latency.py                  per-stage device + latency log
scripts/                      00–22, in workflow order (see docs/RUNBOOK.md)
docs/DEMO.md                  measured results and the placement argument
```

Conventions: nothing scenario-specific is hardcoded — it goes in
`config/scenario.yaml`. Every inference stage logs its device and latency.

---

## Known gaps

Stated plainly rather than implied.

- **The policy is not OpenVINO IR.** The detector is; SmolVLA runs in PyTorch on
  the Arc. `physicalai export` requires a Lightning checkpoint and ours came
  from `lerobot-train`. [scripts/21_export_policy_ov.py](scripts/21_export_policy_ov.py)
  works around that by calling the Policy mixin's `export()` directly and loads
  the trained weights successfully, but stops in
  `_get_default_export_input_sample` — tracing needs an explicit input sample.
- **Policy training used `lerobot-train`, not `physicalai fit`.** The Studio path
  is built and reaches the Lightning trainer
  ([16_studio_train.sh](scripts/16_studio_train.sh)); the demo checkpoint did not
  come from it. The dataset is LeRobot v3 either way, which Studio reads natively
  with no import step.
- **The detector is calibrated to a known brick set.** PatchCore is a memory-bank
  model: a brick colour absent from the normals scores as anomalous. Adding one
  is ~10 crops and a one-minute refit, documented in
  [docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Docs

- [docs/RUNBOOK.md](docs/RUNBOOK.md) — every command, and troubleshooting
- [docs/DEMO.md](docs/DEMO.md) — measured results
- [docs/CHALLENGE.md](docs/CHALLENGE.md) — requirements and rubric
- [docs/SETUP.md](docs/SETUP.md) — the rig and verified environment
- [CLAUDE.md](CLAUDE.md) — working notes and constraints

## License

Apache 2.0 — see [LICENSE](LICENSE).
