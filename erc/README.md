# ERC Workspace

`erc/` is the home for all project-owned code in this repository.

## Structure

- `common/`: shared helpers used across experiments
- `tools/`: environment and local utility scripts
- `telme_sharingan/`: integration code that combines TelME and Sharingan

## Design Rule

- Keep `TelME/` as the upstream MERC baseline.
- Keep `sharingan/` as the upstream gaze model.
- Implement any glue code, adapters, feature builders, training wrappers, and evaluation logic under `erc/`.

This keeps external repositories easy to update and makes the integration layer easy to find.
