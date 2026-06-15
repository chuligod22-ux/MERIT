# Anonymization log — representative / released frames

Checklist applied to every image placed in `data/samples/` (and to any raw frames released
on request). Tick each item before public release.

| # | Check | Status |
|---|-------|--------|
| 1 | EXIF / metadata stripped (GPS, timestamp, device serial, operator) | [ ] |
| 2 | No facility-identifying signage, markers, or labels visible in frame | [ ] |
| 3 | No people, vehicles, or licence plates in frame | [ ] |
| 4 | Frame shows only the chart ROI / lining-crack region of interest | [ ] |
| 5 | Filename carries no identifying information (use neutral IDs) | [ ] |

Notes:
- cam1 frames image an ISO 12233 resolution chart against the lining; cam2 frames image a
  crack-bearing lining region. Both are low-risk (no persons/plates expected), but EXIF
  stripping and a visual pass are still applied.
- Double-blind submission: do **not** publish an author-identifying repository until the
  blind-review requirement is cleared (de-anonymize at acceptance).

EXIF strip example:
```bash
exiftool -all= -overwrite_original data/samples/*.png
```
