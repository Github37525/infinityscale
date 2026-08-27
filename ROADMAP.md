# InfinityScale roadmap

This is a maintainer roadmap, not a promise. Priorities change when real user
feedback, reproducible bugs, or browser constraints change the trade-offs.

## Now: v0.1.x

- Collect reproducible feedback on upload, recommendation, local crop preview,
  repeated processing, stale-result invalidation, and export.
- Fix browser-specific crashes and memory regressions before adding new modes.
- Keep algorithm names, privacy boundaries, and output limitations accurate.

## Next: v0.2.x

- Publish a small, redistributable regression corpus covering photos, line art,
  transparent graphics, text, and gradients.
- Add a documented browser compatibility and memory benchmark.
- Improve the first-run experience when a dependency or model cannot be loaded.

## Later: v0.3.x

- Evaluate a genuinely complete offline distribution, including versioned model
  assets, integrity checks, and an update path.
- Consider batch processing or a command-line route only if recurring user
  feedback shows that the browser workflow is the limiting factor.

## Explicit non-goals

- Calling AI reconstruction or vector tracing “lossless recovery”.
- Uploading source images to an application server for analytics or processing.
- Adding telemetry solely to manufacture adoption metrics.
