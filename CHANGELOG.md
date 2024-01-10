## [Unreleased]

### Added

### Fixed
- Updated docstrings for `run_llama`, `run_chat`, and `run_server` to be more informative.
- Changed default `run_*` parameters to be more sensible for first-time users, including the `run_server` port number (`port=519`) to be unique and not clash with other services.
- Updated run context with the necessary files for fast inference on Metal GPUs (eg, Apple Macbooks M-series)

## [0.2.0]

### Added
- Added `run_server` functionality that starts a simple HTTP server (interact either in your browser or use other LLM packages). It provides an OpenAI-compatible chat completion endpoint.

### Fixed
- Updated llama.cpp JLL bindings to the latest version.
