# Contributing to InfinityScale

Thank you for helping improve InfinityScale.

## Before opening an issue

- Search existing issues first.
- Include the browser name and version, operating system, input dimensions, selected algorithm, scale, and whether post-processing was enabled.
- Do not attach private or copyrighted images unless you have permission to share them. A minimal synthetic or public-domain reproduction is preferred.
- For crashes, include the visible error and relevant console messages, but remove tokens, local paths, and personal information.

## Pull requests

1. Keep changes focused on one problem.
2. Preserve the local-first privacy boundary: image pixels must not be sent to a server.
3. Do not describe AI reconstruction, interpolation, or tracing as lossless recovery.
4. Add or update a reproducible test for behavior changes.
5. Run:

```bash
node --check app.js
node --check sw.js
node tests/smoke.mjs
```

6. Manually verify upload, recommendation, local preview, repeated processing, stale-result invalidation, and the affected export path.

## Algorithm changes

An algorithm change should document:

- Exact upstream package and version.
- Expected image classes and known artifacts.
- Memory and output-size behavior.
- Before/after evidence using distributable test images.
- License compatibility.

## Release expectations

Maintainers should triage issues, review pull requests, keep dependency versions pinned, document security-impacting changes, and publish release notes for user-visible changes.

