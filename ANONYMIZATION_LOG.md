# Anonymization log

## Revision-v2 staging audit

- Four-tunnel image pixels and expert-mask pixels are excluded from the Git tree and all
  planned public release assets.
- Field sites are represented by anonymous tunnel labels A–D and opaque span identifiers.
- Public cohort records replace machine-local image and mask locations with
  `REQUEST_ONLY/...` placeholders.
- Windows and Unix machine-local absolute paths were redacted from copied text artifacts.
- Checkpoints, raw archives, progress shards, caches, PID files, and third-party source
  trees were excluded by the assembly allowlist.
- All 737 cam2 and 479 cam1 public crops were re-encoded without inherited metadata.
  All 50 byte-exact printed-width PNGs had empty PNG metadata-key sets and no EXIF.
- A condition-stratified visual audit covered ten cam1 crops, ten cam2 crops, and ten
  printed-width inputs. The reviewed controlled images contained no people, vehicle
  plates, equipment serials, facility names, route names, location signs, or other
  text-bearing identity panels.
- Release verification confirmed zero four-tunnel image or expert-mask pixels across the
  Git tree and all six staged releases.

The verification script fails if it finds a user-directory absolute path, a prohibited
model binary in the Git tree, or a four-tunnel image/mask asset.
