# Build Plan

Target: a closed loop where the SO-101 follower inspects LEGO bricks, classifies
each defective/good, and places it in the matching plate — with inference running
on the Core Ultra NPU/GPU via OpenVINO.

---

## Guiding decisions (and why)

These are the bets. Revisit them only with a reason.

**1. Scripted pick-and-place, not a learned policy.**
The follower moves via inverse kinematics to coordinates from a calibrated
camera transform. Deterministic, debuggable, no training time. A learned ACT
policy is the *stretch goal* (Phase 9), not the critical path — collecting
demos and training would consume most of Challenge Day, and a policy that
half-works scores worse on the 10-point reliability criterion than a scripted
motion that always works.

**2. Planar homography, not full hand-eye calibration.**
Every brick sits on the flat mat, so Z is constant. A 4-point homography maps
overhead pixels → robot XY. This is minutes of work instead of hours, and is
*more* accurate on a plane than a general hand-eye solve.

**3. Classical segmentation for localization, Anomalib for defect.**
Bright saturated bricks on a dark matte mat is about the easiest segmentation
problem there is — HSV threshold + connected components gives centroid and
orientation with zero training. Do **not** burn Challenge Day training a YOLO
detector to find objects you can threshold. Anomalib is reserved for the actual
defect decision, which is what the rubric scores.

**4. Anomalib with a memory-bank model (PaDiM or PatchCore).**
These fit on *normal samples only* with no backprop — photograph ~50 good
bricks, fit in minutes. Critical because the defect definition arrives at
kickoff and there will be very few defective examples to learn from.

**5. Everything scenario-specific lives in `config/scenario.yaml`.**
Defect criterion, class→plate mapping, brick colors, thresholds, poses. The
kickoff reveal should be a config edit, not a code change.

---

## Phase 0 — Environment sanity · ~30 min

- [ ] Re-run the organizers' preflight script, confirm `FAIL 0`
- [ ] **Fix the `physicalai` import warning** (see SETUP.md) — re-run
      `2_install_software.sh` under `~/physical-ai-studio/application/backend/`
- [ ] Verify OpenVINO sees all three devices:
      `python -c "import openvino as ov; print(ov.Core().available_devices)"`
      → expect `['CPU', 'GPU', 'NPU']`
- [ ] `git clone` this repo on the Ubuntu box

**Gate:** NPU visible to OpenVINO. If not, stop and fix — 20 points ride on it.

## Phase 1 — Hardware bring-up · ~60 min

- [ ] Install / locate LeRobot; identify the USB port for each arm
- [ ] Calibrate **follower** (left, gripper) and **leader** (right, handle)
- [ ] Teleop smoke test — leader drives follower, gripper opens/closes
- [ ] Enumerate cameras: `v4l2-ctl --list-devices`. Record which `/dev/video*`
      is overhead vs. inspect **into `scenario.yaml`** — indices reshuffle on reboot,
      prefer stable `/dev/v4l/by-id/` paths
- [ ] **Reach test** (see SETUP.md risk): jog the follower to both plate positions
      and all four corners of the brick zone. Move the plates inward until every
      target is comfortably inside the envelope. Do this before calibrating.
- [ ] Record and save: home pose, both plate drop poses, safe retreat pose

**Gate:** follower can physically reach every point the task requires.

## Phase 2 — Workspace calibration · ~90 min · ⚠ highest risk

This is the phase teams underestimate and the one that silently breaks
everything downstream.

- [ ] **Rigidly fix the overhead camera.** Tape the boom. Any bump after this
      point invalidates the calibration — if the arm starts missing, suspect
      this first.
- [ ] Collect correspondence pairs: jog the gripper tip to a known mat point,
      record `(pixel_xy, robot_xy)`. Get 6+ spread across the working area,
      not clustered.
- [ ] Solve with `cv2.findHomography`, save to `config/calibration.yaml`
- [ ] **Validate:** click a pixel, command the arm there, measure the error.
      Target **< 5 mm**. Iterate until it holds across the whole mat, including
      corners — error is usually worst at the edges.

**Gate:** click-to-touch accuracy under 5 mm everywhere.

## Phase 3 — Perception · ~90 min

**3a. Brick localization (overhead cam)**
- [ ] HSV threshold → connected components → centroid + orientation per brick
- [ ] Filter blobs by area to reject the plates, the mat edge, and hands
- [ ] Emit a structured list: `[{id, pixel_xy, robot_xy, angle, crop}]`

**3b. Defect detection (Anomalib)**
- [ ] Capture a normal-only dataset — ~50 good bricks, varied color/position/angle
- [ ] Train PaDiM or PatchCore on normals
- [ ] Export to OpenVINO IR
- [ ] Emit `{anomaly_score, heatmap, is_defective}` per brick crop
- [ ] Tune the threshold on a held-out set once the real defect is known

**Gate:** overhead frame in → list of bricks with robot coords and anomaly scores out.

## Phase 4 — VLA / Physical AI Studio · ~90 min

Where the 20-point integration score lives. There is **no worked example of
Anomalib feeding Physical AI Studio** — this wiring is original work, which is
also the innovation angle for the demo.

- [ ] Stand up the Physical AI Studio workflow
- [ ] Define the action/skill interface the robot layer consumes:
      `{action: pick_and_place, brick_id, target_plate, confidence}`
- [ ] Feed the VLA: the task instruction + the structured brick list from Phase 3
- [ ] VLA decides **which plate** each brick goes to and handles ambiguity
- [ ] **Uncertainty band** — scores near the threshold must not be silently
      guessed. Define an explicit behavior (re-inspect from the second camera,
      or route to a third "unsure" position). This is directly scored under
      reliability and is cheap to implement.

**Gate:** brick list in → action intents out, including a defined unsure path.

## Phase 5 — Motion execution · ~60 min

- [ ] Pick sequence: approach above brick → descend → close gripper → lift →
      traverse to plate → release → retreat to home
- [ ] Use the Phase 2 homography for XY; fixed Z heights per stage
- [ ] Align gripper yaw to the brick's detected orientation
- [ ] **Grasp verification** — check gripper position after closing. If it closed
      fully, the grasp missed. Retry once, then skip that brick and continue.
      Never let one failed grasp stall the demo.

**Gate:** commanded action → brick reliably lands in the right plate.

## Phase 6 — OpenVINO optimization · ~60 min

- [ ] Convert the Anomalib model to OpenVINO IR
- [ ] Device placement — start here, then measure and adjust:
      | Stage | Device | Rationale |
      |---|---|---|
      | Anomalib inference | **NPU** | Sustained fixed-shape inference, frees CPU |
      | Any heavier vision model | **GPU** | Arc iGPU, higher throughput |
      | Segmentation, IK, control | **CPU** | Light, latency-sensitive, irregular |
- [ ] Try INT8 quantization; keep it only if accuracy holds
- [ ] Benchmark per-stage and record the numbers — **you must explain these live**
- [ ] Build the latency table for the demo: stage → device → ms

**Gate:** a written table of stage/device/latency, and a reason for each placement.

## Phase 7 — Closed loop + reliability · ~60 min

- [ ] Run the full loop unattended over all bricks
- [ ] Handle: empty workspace (stop cleanly), failed grasp (retry then skip),
      uncertain detection (the Phase 4 unsure path), brick dropped in transit
- [ ] Measure end-to-end perception→action latency
- [ ] Run it 3+ times and record the success rate honestly

**Gate:** loop runs start to finish without a human touching it.

## Phase 8 — Demo prep · ~30 min

- [ ] Architecture diagram: the five stages, with the device for each
- [ ] Latency table from Phase 6
- [ ] Rehearse the explanation: architecture → workload placement → optimization
      choices → observed results. The rubric asks for exactly these four.
- [ ] Reset procedure between runs, so a retry is fast and calm

## Phase 9 — Stretch, only if Phases 0-8 are done

- [ ] Teleop demo collection with the leader arm → train ACT →
      export to OpenVINO ([Intel ships ACT pre-converted](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_act.html))
- [ ] Run the learned policy as an alternate mode, with scripted as fallback

Demoing both, and being able to explain the tradeoff, is a strong innovation story.
Attempting only this and having it fail is the worst outcome on the board.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Camera→robot calibration drifts | Everything misses | Rigidly fix camera; re-validate after any bump; keep the 10-min recalibration script ready |
| Defect definition unknown until kickoff | Detector may not fit | Normal-only Anomalib model; threshold + criterion in config |
| No Anomalib ↔ PAIS example exists | Phase 4 overruns | Define a plain JSON contract between them early; stub it in Phase 4 hour one |
| Plate outside follower's reach | Task impossible | Phase 1 reach test **before** calibration |
| `physicalai` import broken | Blocks Phase 4 | Fix in Phase 0 while there's debug time |
| Lighting shifts at the venue | Both HSV and Anomalib drift | Capture training data under demo lighting; keep HSV bounds in config |
| Single failed grasp stalls the run | Demo dies live | Retry-once-then-skip in Phase 5 |

## Fallback ladder

If time runs short, cut from the bottom. Never cut the loop.

1. **Non-negotiable** — closed loop, one brick, perception → action. *Without
   this there is no submission.*
2. Anomalib on NPU via OpenVINO with recorded latency
3. VLA reasoning through Physical AI Studio
4. Uncertainty handling + grasp retry
5. INT8 quantization
6. Multi-brick continuous sorting
7. Learned ACT policy

A scripted loop that sorts one brick correctly, end to end, beats every
partially-integrated alternative. Get to step 1 early, then climb.
