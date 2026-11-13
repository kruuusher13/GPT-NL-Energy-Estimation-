#!/usr/bin/env bash
# Rebuild the notebook + the interactive slide deck in one go, with the fit-CSS that
# guarantees every figure/table/screenshot fits the reveal.js slide canvas.
set -e
cd "$(dirname "$0")"

echo "[1/4] building notebook structure ..."
python3 build_presentation.py
# ensure the trained-model artifacts exist (the notebook loads them back)
[ -f models_manifest.json ] || { echo "      training + saving models (first run) ..."; python3 train_and_save.py >/dev/null; }

echo "[2/4] executing (real data, plots, tables) ..."
python3 -m nbconvert --to notebook --execute --inplace GPTNL_Energy_Notebook.ipynb \
        --ExecutePreprocessor.timeout=300 >/dev/null

echo "[3/4] exporting figure-per-slide deck (code hidden so graphs always render) ..."
python3 -m nbconvert --to slides --no-input GPTNL_Energy_Notebook.ipynb >/dev/null
python3 -m nbconvert --to html GPTNL_Energy_Notebook.ipynb >/dev/null   # full reading copy WITH code

echo "[4/4] injecting fit-CSS into the deck ..."
python3 - <<'PY'
f="GPTNL_Energy_Notebook.slides.html"; h=open(f).read()
css="""<style>
.reveal .slides{font-size:24px;}
.reveal h1{font-size:1.6em;} .reveal h2{font-size:1.3em;} .reveal h3{font-size:1.1em;}
.reveal p,.reveal li{font-size:0.82em; line-height:1.25;}
.reveal img{max-height:540px; max-width:92%; height:auto; width:auto; display:block; margin:0.2em auto;}
.reveal table{font-size:0.46em; margin:0 auto; border-collapse:collapse;}
.reveal pre,.reveal code{font-size:0.5em;}
.reveal blockquote{font-size:0.8em; width:90%;}
.reveal section{overflow-y:auto; max-height:96vh;}
</style>"""
open(f,"w").write(h.replace("</head>", css+"\n</head>",1)); print("  fit-CSS injected")
PY
echo "DONE -> GPTNL_Energy_Notebook.slides.html  (open it; arrows navigate, Esc = overview)"
