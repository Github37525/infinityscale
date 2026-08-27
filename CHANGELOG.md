# Changelog

All notable changes to InfinityScale are documented here.

## [0.1.0] - 2026-08-27

### Added

- Local-first upload, recommendation, crop selection, local preview, and export workflow.
- VTracer SVG tracing with persistent WebAssembly worker.
- ESRGAN Thick 2× and 4× model paths through UpscalerJS.
- Pica MKS2013 high-quality resampling.
- PNG, SVG, and print-layout PDF export.
- Output scale, predicted dimensions, target DPI, and effective-DPI feedback.
- PWA shell and responsive desktop/mobile interface.

### Fixed

- Consecutive processing and tensor/model cleanup.
- Stale results remaining exportable after parameter changes.
- Large temporary export canvases and avoidable base64 copies.
- Overstated lossless, offline, and print-quality claims.

