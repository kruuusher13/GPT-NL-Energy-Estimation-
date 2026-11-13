#!/usr/bin/env python3
"""Fix the broken venv_path in the GPT-NL pipeline.

stage.py / deduplication.py set:
    venv_path = Path(__file__).parents[N] / "venv" / "bin" / "activate"
which points at a local ./venv that does not exist (the project uses Poetry).
The env_command then runs `module load Python; source <nonexistent>/activate`,
so after `module load` shadows PATH the venv is never restored and the
merge_stats / dependent jobs hit `command not found`.

Fix: point venv_path at the ACTUAL active venv via sys.executable (resolved at
submit time into a concrete absolute path written into the sbatch script).
Idempotent.
"""
import re
import sys

OLD = re.compile(r'Path\(__file__\)\.parents\[\d+\]\s*/\s*"venv"\s*/\s*"bin"\s*/\s*"activate"')
NEW = 'Path(sys.executable).resolve().parent / "activate"'

for path in sys.argv[1:]:
    s = open(path).read()
    changed = False

    if "import sys" not in s:
        # add `import sys` right after the first `import os`
        s = s.replace("import os\n", "import os\nimport sys\n", 1)
        changed = True

    if OLD.search(s):
        s = OLD.sub(NEW, s)
        changed = True

    if changed:
        open(path, "w").write(s)
        print(f"patched {path}")
    else:
        print(f"no change needed {path}")
