import glob, statistics as st
import pyarrow.parquet as pq

root = "/projects/0/prjs0986/energy_studies/datasets"
# all 16 corpora in Julio's datasets/ — probe each for OOM-safe sizing + content features
cands = ["american-stories","european_parliament","cc_github_opencode","cc_german-pd",
         "cc_english_pd","kb-pd-books","rechtspraak","cc_openalex","auditdienstrijk",
         "belgian-journal","dansknaw","dpc","naturalis","tweedekamer","utrechtsarchief","woogle"]

print(f"{'corpus':<20} {'files':>6} {'avg_chars':>10} {'max_chars':>10} {'sym%':>6} {'avgwd':>6}  textcol")
for c in cands:
    fs = sorted(glob.glob(f"{root}/{c}/*.parquet")) or \
         sorted(glob.glob(f"{root}/{c}/**/*.parquet", recursive=True))
    if not fs:
        print(f"{c:<20} {'NO PARQUET':>10}"); continue
    try:
        pf = pq.ParquetFile(fs[0]); cols = pf.schema.names
        tc = "text" if "text" in cols else cols[0]
        sample = pf.read_row_group(0, columns=[tc]).column(0).to_pylist()[:300]
        txt = [x for x in sample if isinstance(x, str) and x]
        lens = [len(x) for x in txt]
        # content features: non-alnum fraction, avg word length
        sym = st.mean([sum(1 for ch in x if not ch.isalnum() and not ch.isspace())/max(len(x),1)
                       for x in txt]) if txt else 0
        awl = st.mean([st.mean([len(w) for w in x.split()]) if x.split() else 0 for x in txt]) if txt else 0
        avg = int(st.mean(lens)) if lens else 0; mx = max(lens) if lens else 0
        print(f"{c:<20} {len(fs):>6} {avg:>10} {mx:>10} {sym*100:>5.1f}% {awl:>6.1f}  {tc}")
    except Exception as e:
        print(f"{c:<20} ERROR {type(e).__name__}: {str(e)[:40]}")
