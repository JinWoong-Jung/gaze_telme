# TelME + Sharingan Integration

This directory is the implementation area for project-specific code that combines:

- `TelME/`: MERC baseline model and training code
- `sharingan/`: gaze-following model and upstream utilities

Planned responsibilities here:

- dataset adapters that align MELD samples with gaze features
- preprocessing and feature-building code for utterance-level gaze vectors
- model wrappers that inject gaze into the TelME video/fusion pipeline
- evaluation and analysis code for the integrated system

Upstream repositories should stay as untouched as possible. Local glue code belongs here.
