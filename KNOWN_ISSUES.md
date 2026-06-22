# Known Issues / Tech Debt

## Cross-repo dependency on the retired `nzgd_data_extraction` package

`nzgd/scripts/extract/bh/extract_ground_water_level_from_borehole_ags.py` imports
from the legacy `nzgd_data_extraction` package:

```python
from nzgd_data_extraction import info          # line 14
...
encoding = info.find_encoding(ags_file)         # line 104 — the only use
```

This is the **sole remaining code-level coupling** from the active `nzgd` repo to
the retired `nzgd_data_extraction` repo. The only other mentions are prose, not
dependencies: a docstring credit in `nzgd/metadata/location.py` and the design
docs under `docs/superpowers/`.

Notes for whoever handles this later:

- `nzgd_data_extraction` is **not installed in `dev_nzgd_venv`** — importing it
  raises `ModuleNotFoundError` — so this borehole groundwater-level extraction
  script currently cannot run in that environment as-is.
- The dependency is a single helper: `info.find_encoding(...)` (file-encoding
  detection). To fully decouple, vendor that one function into the `nzgd` package
  (e.g. a small `nzgd/extract/bh/encoding.py`) and drop the import. After that,
  `nzgd_data_extraction` can be retired without affecting the active pipeline.

Discovered 2026-06-22 during the regenerated-NZGD-index work.
