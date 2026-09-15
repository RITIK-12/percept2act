# percept2act

**Perceive → Detect → Reason → Act → Optimize.**

An SO-101 robotic arm that inspects LEGO bricks, classifies each as defective or
good, and sorts it into the matching plate — with inference optimized by OpenVINO
and running on an Intel Core Ultra Series 3 NPU and iGPU.

Built for the Intel Physical AI Challenge.

## Stack

| Layer | Component |
|---|---|
| Arm | SO-101 (leader/follower pair, LeRobot) |
| Localization | Classical HSV segmentation |
| Defect detection | Anomalib |
| Reasoning | Intel Physical AI Studio (VLA) |
| Optimization | OpenVINO — NPU / GPU / CPU |

## Docs

- **[CLAUDE.md](CLAUDE.md)** — start here
- [docs/PLAN.md](docs/PLAN.md) — build plan, phases, fallback ladder
- [docs/SETUP.md](docs/SETUP.md) — the rig and verified environment
- [docs/CHALLENGE.md](docs/CHALLENGE.md) — requirements and rubric
- [docs/LINKS.md](docs/LINKS.md) — verified reference documentation
- [config/scenario.yaml](config/scenario.yaml) — all scenario-specific values

## Status

Planning. No implementation yet — see [docs/PLAN.md](docs/PLAN.md) Phase 0.

## License

Apache 2.0 — see [LICENSE](LICENSE).
