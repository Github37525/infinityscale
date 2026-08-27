# Security Policy

## Supported version

Security fixes are applied to the latest release on the default branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this repository. Do not open a public issue for an unpatched vulnerability.

Include:

- Affected browser and release.
- Reproduction steps and impact.
- Whether the issue can expose image data, execute untrusted script, poison cached assets, or cause persistent denial of service.
- A minimal proof of concept that does not include private images or credentials.

Maintainers will acknowledge a complete report within seven days and will coordinate remediation and disclosure based on severity.

## Security boundaries

- Selected image pixels are intended to stay in the browser.
- Runtime dependencies and AI model files are fetched from pinned public CDN URLs.
- Exported files are generated locally.
- This project does not provide an authentication or cloud-storage service.

If a contribution changes any of these boundaries, it must update the README and receive explicit maintainer review.

