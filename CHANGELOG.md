# Changelog

All notable changes to the NeoDEM training worker are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning uses [CalVer](https://calver.org/) (`YYYY.MM.DD`) for daily releases.

## [v2026.07.11] - 2026-07-11

### Added

- LeRobot 0.6.0 — reward-model/annotate job kinds, native GR00T N1.7 trainer, GPU setup (#5)

### Fixed

- GR00T live loss telemetry, parallel S3 I/O, slim artifacts, Windows-safe cancel (#7)


## [v2026.07.04] - 2026-07-04

### Added

- **GR00T N1.x fine-tuning trainer** via the NVIDIA Isaac-GR00T subprocess. Jobs whose `baseModel` contains `gr00t` fine-tune GR00T N1.7/N1.5: `worker.py` dispatches the trainer per job, `trainers/gr00t_n1.py` runs `launch_finetune.py` in Isaac-GR00T's own uv env (CUDA), converts LeRobot v3→v2, auto-generates the modality config (overridable via `modality_config_path`, e.g. `UNITREE_G1_SONIC` whole-body), streams step/loss progress, honours cancel by killing the process group, and uploads the newest checkpoint. Covered by 35 GPU-free tests; ready for real Unitree G1 validation (`docs/gr00t-unitree-g1.md`).

### Maintenance

- CalVer release automation — an always-open Release PR (`prepare-release-pr.yml`) that on merge tags `vYYYY.MM.DD`, publishes a GitHub Release, and builds/pushes `ghcr.io/raasaar-org/neodem-training-worker`.


## [v2026.04.12] - 2026-04-12

### Added

- Initial release of the NeoDEM training worker, extracted from robot-management-system (TASK-150). Polls server for jobs, runs SmolVLA LoRA fine-tuning, streams progress back over HTTP.
