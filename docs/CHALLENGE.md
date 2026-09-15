# Challenge Requirements & Rubric

Summarized from the Intel Physical AI Challenge technical brief (4pp).
The PDF is gitignored; this is the working reference.

---

## The ask

Build an end-to-end Physical AI solution where an SO-101 arm performs autonomous
defect detection **and the appropriate physical response**.

The brief states the core requirement twice, in different words: the final system
must **connect defect detection to an autonomous robot action**, not demonstrate
perception as an isolated model. That is the trap to avoid.

## Mandated stack — no substitutions

| Layer | Component |
|---|---|
| Arm | SO-101 |
| Orchestration / VLA | Intel Physical AI Studio |
| Anomaly detection | Anomalib |
| Optimization | OpenVINO |
| Silicon | Intel Core Ultra Series 3 — NPU + GPU + CPU |

## Required pipeline

```
PERCEIVE  →  DETECT   →  REASON    →  ACT      →  OPTIMIZE
camera       Anomalib     VLA in       SO-101      OpenVINO on
input        score +      Physical     executes    Core Ultra 3
             localization AI Studio    the place
```

## Judging rubric — 100 points

| Criterion | Points | What it rewards |
|---|---|---|
| End-to-end Physical AI solution | **25** | One integrated closed loop, *not* independent component demos |
| Defect detection with Anomalib | **20** | Detection quality **and** how defect output is exposed to the robotics workflow |
| VLA & Physical AI Studio integration | **20** | Perception + instruction → appropriate robot action |
| OpenVINO & Core Ultra 3 optimization | **20** | Latency, throughput, appropriate NPU/GPU/CPU placement |
| Robotic execution & reliability | **10** | Accuracy, repeatability, **recovery from uncertain detections** |
| Innovation & demonstration | **5** | Originality, clarity of the live explanation |

### Reading the rubric

- **65 points are integration.** End-to-end (25) + VLA glue (20) + deployment (20)
  all reward plumbing. Raw detector accuracy is only a slice of the 20-point
  Anomalib bucket. Build the loop first.
- **Recovery behavior is explicitly scored** under reliability. An uncertainty
  threshold with a defined fallback is cheap to build and most teams skip it.
- **You must explain workload placement out loud.** Log per-stage device and
  latency from day one rather than reconstructing numbers at the end.

## Deliverables

- Working closed-loop prototype, perception → action
- VLA + Anomalib integration
- OpenVINO-optimized deployment on Core Ultra 3
- Live demo + architecture/optimization explanation

## ⚠ Revealed only at kickoff

The defect definition, target objects, task rules, and operating conditions are
**withheld until Challenge Day**.

This is the single biggest architectural constraint in the brief. Every
scenario-specific value must be swappable config, never hardcoded.
See [../config/scenario.yaml](../config/scenario.yaml).
