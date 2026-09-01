# Security policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

| Version | Supported |
|---|---|
| latest release (0.1.x) | yes |
| older | no |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository:
[Report a vulnerability](https://github.com/majmichu1/Astraios/security/advisories/new).
It is enabled and goes only to the maintainer. Please do not open a public
issue for security problems. There is no separate security email address.

## What Astraios does with your data

- Image processing and AI inference run on your machine. No telemetry,
  analytics or crash reports are collected, and there is no account or
  licence system.
- Network access is on demand only:
  - startup update check against `api.github.com` (Preferences > Updates to disable);
  - downloads of the Real-ESRGAN super-resolution weights (GitHub), Gaia
    catalog files and minor-body orbital elements (Seti Astro data) when you
    use those features;
  - catalog queries to SIMBAD, VizieR and hips2fits (CDS) when you identify
    objects or run the Smart Processor;
  - the astrometry.net solver, which uploads a 16-bit mono FITS copy of the
    image being solved, and only when you select that backend and enter an
    API key. ASTAP and the built-in local Gaia solver work offline.
- The full list with file references is in
  [docs/project-facts.md](docs/project-facts.md#local-processing-network-use-uploads-telemetry).

## Downloads and integrity

- The bundled AI denoise model ships inside the package. The Real-ESRGAN
  weights are downloaded from the upstream GitHub release on first use and
  stored in the user cache.
- Model files fetched through the model manager are verified against a
  SHA-256 hash before use; a model without a published hash is never
  downloaded.
- The updater verifies a downloaded installer against the `.sha256` sidecar
  published with the release when one is present. Releases built from
  `.github/workflows/build.yml` publish one sidecar per asset.
- StarNet, Cosmic Clarity and other third-party models or binaries are only
  used if you install them and point Astraios at them in Preferences.
