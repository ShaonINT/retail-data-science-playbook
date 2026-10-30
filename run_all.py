"""Regenerate every case-study table and chart in The Retail Data Science Playbook.

Usage:  python run_all.py            # all case studies
        python run_all.py ch09 ch10  # selected chapters
Results are written to results/*.json and charts to figures/*.svg.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = sorted((ROOT / "case_studies").glob("ch*.py"))
wanted = sys.argv[1:]
if wanted:
    scripts = [s for s in scripts if any(s.name.startswith(w) for w in wanted)]
failed = []
for s in scripts:
    t = time.time()
    print(f"▶ {s.name}", flush=True)
    r = subprocess.run([sys.executable, str(s)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        failed.append(s.name)
        print(r.stderr[-2000:])
    print(f"  {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f}s")
print(f"\n{len(scripts) - len(failed)}/{len(scripts)} case studies reproduced.")
sys.exit(1 if failed else 0)
