"""Kaggle CPU run (no GPU hours): how fast does Kaggle download PAR slides from the EBI server?

Downloads 6 Grundium slides, first one at a time, then 3 at once, and prints MB/s. This
decides whether the PAR feature run can download while it works without wasting GPU time.
Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/par_speedtest
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

B = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/323/S-BIAD2323/Files/bia_prostate_wsi/component_A_grundium_svs/"
SLIDES = ["c010a", "c010b", "c011a", "c011b", "c012a", "c012b"]   # names checked against kaggle/par/grundium_urls.txt


def get(name):
    t = time.time()
    n = 0
    try:
        with urllib.request.urlopen(B + f"{name}_grundium.svs", timeout=120) as r:
            while chunk := r.read(1 << 22):
                n += len(chunk)
    except OSError as e:
        print(f"{name}: failed ({e})", flush=True)
    sec = time.time() - t
    print(f"{name}: {n / 1e6:.0f} MB in {sec:.0f} s = {n / 1e6 / max(sec, 1e-9):.1f} MB/s", flush=True)
    return n, sec


res = {}
t = time.time()
one = [get(s) for s in SLIDES[:3]]
res["one_at_a_time_MBps"] = round(sum(n for n, _ in one) / 1e6 / (time.time() - t), 1)
t = time.time()
with ThreadPoolExecutor(3) as ex:
    three = list(ex.map(get, SLIDES[3:]))
res["three_at_once_MBps"] = round(sum(n for n, _ in three) / 1e6 / (time.time() - t), 1)
res["per_slide"] = [(s, round(n / 1e6), round(sec)) for s, (n, sec) in zip(SLIDES, one + three)]
print(json.dumps(res, indent=1))
json.dump(res, open("/kaggle/working/speed.json", "w"), indent=1)
