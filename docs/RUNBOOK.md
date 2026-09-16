# Runbook

Every command copy-paste runnable. `conda activate hack_lerobot && cd ~/percept2act` first.

## The demo

```bash
python scripts/12_demo.py
```

Camera watches the inspect spot. Put a brick down; it settles, PatchCore scores
it on the NPU, the verdict picks an instruction, the arm places it in the
matching plate and returns. Repeat. `q` or Ctrl-C to stop.

| flag | |
|---|---|
| `--bricks N` | stop after N (default: until Ctrl-C) |
| `--delay S` | seconds to hold the verdict on screen before moving (default 2) |
| `--settle N` | frames the scene must hold steady before acting (default 12) |
| `--executor stub` | run the perception only, arm never moves |
| `--executor policy` | use the fine-tuned SmolVLA instead of replay |
| `--device GPU` | run the detector somewhere else |

Place the brick where it sat during recording: `replay` repeats joint angles and
does not look at the brick. That constraint is exactly what the SmolVLA policy
removes.

---

## Scripts, in workflow order

| | | |
|---|---|---|
| `00_preflight.sh` | check NPU, arms, cameras, calibration | run in every new shell |
| `01_scan_motors.py` | diagnose "Missing motor IDs" | when an arm drops off |
| `02_assign_cameras.py` | map RealSense serials to roles | after a camera replug |
| `03_camera_focus.py` | live view with a sharpness meter | focusing a lens |
| `04_tune_exposure.py` | sweep and save wrist exposure | lighting changed |
| `05_set_inspect_pose.py` | teleop to a pose + crop, save both | rig moved |
| `06_record.sh` | record demos: `coral N` / `blue N` | |
| `07_verify_dataset.sh` | check both instructions, balanced | after recording |
| `08_hold_pose.py` | park at the inspect pose and hold | while capturing bricks |
| `09_capture_bricks.py` | capture crops: `--class good\|defective` | |
| `10_train_detector.py` | fit PatchCore, export OpenVINO IR | |
| `11_benchmark.py` | device sweep + score separation | |
| `12_demo.py` | **the live demo** | |
| `13_start_studio.sh` | Physical AI Studio backend / ui | |
| `14_train_vla.sh` | fine-tune SmolVLA via `lerobot-train` | torch only |
| `15_studio_record.sh` | record through Studio + verify | |
| `16_studio_train.sh` | fine-tune SmolVLA via `physicalai fit` | exportable |
| `17_export_policy.sh` | export the policy to OpenVINO IR | needs 16 |
| `19_record_clean.py` | record with a fixed start pose | **use this to re-record** |

---

## Rebuilding the detector

```bash
python scripts/08_hold_pose.py --from-config
```

```bash
python scripts/09_capture_bricks.py --class good --auto 50 --interval 1.0
```

```bash
python scripts/09_capture_bricks.py --class defective --auto 15
```

```bash
python scripts/10_train_detector.py && python scripts/11_benchmark.py
```

Put the threshold it reports into `detector.threshold`. Brick condition matters
here and nowhere else.

---

## Recording more demos

```bash
bash scripts/06_record.sh coral 10
```

```bash
bash scripts/06_record.sh blue 10
```

Right arrow ends an episode, left arrow redoes it, Escape stops and saves. Not
Ctrl-C — that loses the take. The argument names the **destination plate**, not
the brick's condition; use any bricks.

```bash
bash scripts/07_verify_dataset.sh
```

---

## Fine-tuning the VLA

```bash
bash scripts/14_train_vla.sh --steps=8000
```

Any flag passes through to `lerobot-train`. Batch size does not buy speed — the
iGPU is compute-bound, so `--batch_size=16` roughly doubles per-step time and
`bs16 x 4000` sees the same samples as `bs8 x 8000` in about the same wall
clock. Cut `--steps` if you need it shorter.

Then swap one flag:

```bash
python scripts/12_demo.py --executor policy
```

### Two training paths, and why it matters

| | checkpoint | runs in |
|---|---|---|
| `14_train_vla.sh` (`lerobot-train`) | `pretrained_model/model.safetensors` | torch-xpu only |
| `16_studio_train.sh` (`physicalai fit`) | Lightning `.ckpt` | torch **or** OpenVINO |

`physicalai export` loads via Lightning's `load_from_checkpoint`, so only the
second format can be exported. Train through Studio if you want the policy on
OpenVINO rather than torch:

```bash
bash scripts/16_studio_train.sh && bash scripts/17_export_policy.sh
```

SmolVLA's OpenVINO export includes the tokenizer, so the instruction string is
tokenized by OpenVINO too — the whole Anomalib → instruction → policy seam runs
on OpenVINO runtime.

---

## When something breaks

| Symptom | Fix |
|---|---|
| Arm port permission denied | `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1` |
| "Missing motor IDs" | `python scripts/01_scan_motors.py` — usually an overload latch; unplug the arm's **power** for 5s |
| Recording stalls on "Sender has been blocked" | stale rerun viewer: `pkill -f rerun`, or `DISPLAY_DATA=false` |
| Wrist camera black | lens cap, or exposure reset on replug — `00_preflight.sh` reapplies it |
| Every score identical | crop is empty or off-target — `05_set_inspect_pose.py` |
| Arm goes to the wrong plate | `replay.episodes` indices — check order with `07_verify_dataset.sh` |
| `import physicalai` fails | `~/hackathon_install/2_install_software.sh 5` |

---

## Verified rig facts

| | |
|---|---|
| Follower | `/dev/ttyACM1`, id `hack_follower` |
| Leader | `/dev/ttyACM0`, id `hack_leader` |
| Wrist camera | Sonix UVC `/dev/video12`, exposure 60, brightness 32 |
| RealSense (unused) | `243722062690` front, `243622062278` boom |
| OpenVINO | `['CPU', 'GPU', 'NPU']` |
| Runtime env | `hack_lerobot` |
| Studio env | `~/physical-ai-studio/application/backend/.venv` |
