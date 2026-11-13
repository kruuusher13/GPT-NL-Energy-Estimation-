#!/usr/bin/env python3
"""Register toxic_language_detection in the pipeline stage registry.
The class is already imported in stages/__init__.py; it's just missing from the
`stages` dict. Idempotent + makes a .bak_toxicreg backup."""
import shutil, sys

F = ("/projects/0/prjs0986/energy_studies/test_romir/data-curation-pipeline/"
     "gptnl_data_curation_pipeline/stages/__init__.py")
s = open(F).read()

if '"toxic_language_detection": {' in s:
    print("[patch] already registered — no change")
    sys.exit(0)

entry = (
    '    "toxic_language_detection": {\n'
    '        "inst": ToxicLanguageDetectionStage(),\n'
    '        "default_input_folder": "test-data/3. heuristically filtered",\n'
    '        "default_output_folder": "test-data/5. toxic language",\n'
    '        "default_logs_folder": "logs/toxic_language_detection",\n'
    '    },\n'
)
needle = ('        "default_logs_folder": "logs/deduplication",\n'
          '    },\n')
if needle not in s:
    print("[patch] ERROR: dedup anchor not found; not patching", file=sys.stderr)
    sys.exit(1)

shutil.copy(F, F + ".bak_toxicreg")
s = s.replace(needle, needle + entry, 1)
open(F, "w").write(s)
print("[patch] registered toxic_language_detection (backup: .bak_toxicreg)")
