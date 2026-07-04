# Changelog

All notable changes to the NeoDEM training worker are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning uses [CalVer](https://calver.org/) (`YYYY.MM.DD`) for daily releases.

## [v2026.04.12] - 2026-04-12

### Added

- Initial release of the NeoDEM training worker, extracted from robot-management-system (TASK-150). Polls server for jobs, runs SmolVLA LoRA fine-tuning, streams progress back over HTTP.
