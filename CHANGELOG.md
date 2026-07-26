# Changelog

All notable changes to CARDAL will be documented in this file.

The format follows [Keep a Changelog], and this project adheres to
[Semantic Versioning].

## [Unreleased]

### Added

- Added signed low-rank objective and constraint matrices, including hybrid
  sparse-plus-low-rank data, across the GPU solver, scaling, rank
  augmentation, multi-GPU execution, public C API, and Python API.
- Added Python `LowRank` and `SparseLowRank` matrix inputs with support for
  diagonal weights or a small symmetric core matrix.
- Added native parsing of SDPT3 low-rank constraints stored in `blk{j,3}` and
  `At{j,2:3}`.
- Added batched GPU kernels for signed low-rank operations on groups of small
  PSD cones, including cones with different adaptive ranks.

## [0.1.1] - 2026-07-25

### Added

- Added a GitHub Pages documentation site covering installation, interfaces,
  parameters, multi-GPU execution, file formats, and result handling.
- Added CLI output files for the low-rank primal factor, per-cone rank list,
  and dual solution alongside the solve summary.
- Added `Model.read_file(path)` as a classmethod that constructs a loaded
  Python model, plus `Model.load_file(path)` for replacing an existing model's
  problem.
- Exposed preconditioning and scaling controls through the public C and Python
  APIs.
- Added native unrestricted real variables across the solver core, SeDuMi and
  SDPT3 readers, C API, and Python API.
- Added explicit C, Python, and CLI result outputs for nonnegative LP primal
  variables.

### Changed

- Standardized CLI long-option names on hyphen-separated spelling.
- MAT readers now reject unsupported SeDuMi and SDPT3 cones, complex data,
  nonlinear barrier objectives, and low-rank SDPT3 encodings instead of
  silently dropping them.

## [0.1.0] - 2026-07-22

- Published the first PyPI source distribution.
- Replaced the shared feasibility tolerance with independent primal and dual
  residual tolerances across the CLI, C API, and Python API.
- Prepared repository metadata, CI, and community documents for public release.
- Added Apache-2.0 source headers across tracked source files.

## [0.0.2] - 2026-07-22

- Fixed Python installation from the source distribution under pip build
  isolation.

## [0.0.1] - 2026-07-14

- Initial public source snapshot of CARDAL.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[Unreleased]: https://github.com/Lhongpei/CARDAL/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Lhongpei/CARDAL/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Lhongpei/CARDAL/releases/tag/v0.1.0
