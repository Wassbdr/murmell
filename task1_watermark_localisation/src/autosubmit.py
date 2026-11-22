"""Candidate check for autonomous submission (rules in SUBMISSIONS.md). Never prints credentials.
    python -m src.autosubmit --status                 # list candidates and whether each condition holds
    python -m src.autosubmit --send <name>            # re-checks everything, then uploads (only if all conditions hold)
    python -m src.autosubmit --verify <name>          # query server for the recorded submission (best effort)
State: work/submit_log.json (sent files: name, file, md5, time, id, server body).
Candidate <name> requires: work/archive/<name>/spec.json (file + md5), work/nested/compare_<name>_vs_<ref>.json with gate
true where <ref> = sent candidate with best nested CV, and work/review/<name>.verdict whose first word is PASS.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from src.data import ROOT

LOG = ROOT / "work" / "submit_log.json"
CADENCE_S = 10 * 60   # human message 21 Nov ~19:05: "la cadence c'est toutes les 10 minutes"
TASK_ID = "34-watermark-localisation-image"


def env():
    """load ~/.wmloc_env (KEY=VALUE lines, optional 'export ') without printing anything."""
    p = os.path.expanduser("~/.wmloc_env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ.get("WMLOC_API_KEY"), os.environ.get("WMLOC_BASE_URL")


def log():
    return json.loads(LOG.read_text()) if LOG.exists() else []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def nested_cv(name):
    p = ROOT / "work/nested" / f"{name}.json"
    return json.loads(p.read_text())["cv_foldmean"] if p.exists() and p.stat().st_size > 0 else None


def is_clean(name):
    p = ROOT / "work/nested" / f"{name}.json"
    return bool(p.exists() and p.stat().st_size > 0 and json.loads(p.read_text()).get("clean", True))


REF_NAME = "S11"      # human 22 Nov 06:55: send the best clean candidate of a pass if it beats S11 in nested CV, or if it is
NEAR = 0.002          # within 0.002 of it and clearly different from every previous send (a weaker send costs nothing: best private counts)
BUG_FLOOR = 0.775     # bug alarm (replaces the "two sends without public rise" watch): a public score below this stops auto-sending
STOP = ROOT / "work/STOP_AUTOSEND"
ALERT = ROOT / "work/ALERT.txt"          # inside the task folder (the guard forbids writing elsewhere)


def alert(msg):
    STOP.touch()
    try:
        with open(ALERT, "a") as f:
            f.write(f"{datetime.now():%F %T} {msg}\n")
    except OSError:
        pass
    print("ALERT", msg)


def final_runs(name, depth=0):
    """runs actually used for the TEST prediction of a candidate (finals of the base + selector sources on all folds)"""
    q = ROOT / "work/nested" / f"{name}.json"
    if depth > 3 or not q.exists() or q.stat().st_size == 0:
        return set()
    d = json.loads(q.read_text()); out = set()
    for v in (d.get("final") or {}).values():
        out |= set(v[0])
    for v in (d.get("sources") or {}).values():
        out |= set(v.get("all", [])) if isinstance(v, dict) else set()
    if d.get("base") and not str(d["base"]).startswith("work/"):
        out |= final_runs(d["base"], depth + 1)
    return out


def uses_sam(name):
    return sorted(r for r in final_runs(name) if "_sam" in r or r.endswith("_ens"))


def differs_clearly(name):
    """>= 5 % of the test images have a mask differing on > 2 % of the cells from EVERY previously sent candidate"""
    import numpy as np
    q = ROOT / "work/test" / f"{name}_bin.npy"
    if not q.exists() or q.stat().st_size == 0:
        return False
    a = np.load(q).astype(bool); worst = 1.0
    for e in log():
        r = ROOT / "work/test" / f"{e['name']}_bin.npy"
        if e["name"] == name or not r.exists() or r.stat().st_size == 0:
            continue
        b = np.load(r).astype(bool)
        if b.shape != a.shape:
            continue
        worst = min(worst, float(((a != b).reshape(len(a), -1).mean(1) > 0.02).mean()))
    return worst >= 0.05


def check(name):
    L = log(); out = {}
    arch = ROOT / "work/archive" / name / "spec.json"
    if not arch.exists() or arch.stat().st_size == 0:      # empty = pre-created placeholder (inode quota)
        return {"archive": False}
    a = json.loads(arch.read_text()); f = ROOT / a["submission_file"]
    out["file"] = str(f.relative_to(ROOT)); out["md5_ok"] = md5(f) == a["submission_md5"]
    out["not_sent"] = all(e["md5"] != a["submission_md5"] for e in L)
    sent = [e for e in L if nested_cv(e["name"]) is not None and is_clean(e["name"])]   # human 05:50: the reference is the best CLEAN send, as in the dispatcher
    ref = max(sent, key=lambda e: nested_cv(e["name"]))["name"] if sent else None
    out["ref"] = ref
    # RULE (human ~19:20): send when the CLEAN nested CV beats the best nested CV already sent and the checker passes;
    # 10-min cadence; never resend an md5. No review / gate / approval needed any more. Leak-inflated CVs are excluded
    # via work/nested/<name>.json key "clean": false (e.g. C4).
    best_sent = max([nested_cv(e["name"]) for e in sent], default=0.0)
    cv = nested_cv(name); meta_ = json.loads((ROOT / "work/nested" / f"{name}.json").read_text()) if cv is not None else {}
    out["nested_cv"] = cv; out["best_sent"] = best_sent
    ref_cv = nested_cv(REF_NAME) or best_sent; out["ref_cv"] = ref_cv
    clean = bool(cv is not None and meta_.get("clean", True))
    out["better"] = bool(clean and (cv > ref_cv or (cv > ref_cv - NEAR and differs_clearly(name))))
    sam = uses_sam(name) if cv is not None else []
    out["sam_runs"] = len(sam); out["sam_ok"] = (not sam) or (ROOT / "work/SAM_CLEARED").exists()   # external pretrained model: wait for the human's clearance
    last = max([e["time"] for e in L], default=0)
    out["wait_s"] = max(0, int(last + CADENCE_S - time.time())); out["cadence_ok"] = out["wait_s"] == 0
    r = subprocess.run(["python", "-m", "src.submit", "--check", str(f)], cwd=ROOT, capture_output=True, text=True)
    out["checker"] = r.returncode == 0 and "VALID" in r.stdout
    out["READY"] = all(out[k] for k in ("md5_ok", "not_sent", "better", "cadence_ok", "checker", "sam_ok"))
    return out


def send(name):
    import requests
    c = check(name)
    if not c.get("READY"):
        print("NOT READY", c); return
    key, base = env()
    if not key or not base:
        print("credentials missing"); return
    a = json.loads((ROOT / "work/archive" / name / "spec.json").read_text()); f = ROOT / a["submission_file"]
    t0 = time.time(); entry = {"name": name, "file": a["submission_file"], "md5": a["submission_md5"], "time": t0,
                               "time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(f, "rb") as fh:
            r = requests.post(f"{base}/submit/{TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (f.name, fh, "application/octet-stream")}, timeout=(10, 180))
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text[:300]}
        entry.update(status=r.status_code, body=body, id=body.get("submission_id") if isinstance(body, dict) else None)
    except requests.exceptions.Timeout as e:
        entry.update(status="TIMEOUT", body=repr(e)[:200], id=None)   # check server-side before any retry
    except requests.exceptions.RequestException as e:
        entry.update(status="ERROR", body=repr(e)[:200], id=None)
    if entry.get("status") == 429:      # rate-limited by the server: NOT a submission, do not log
        print("RATE-LIMITED, not logged:", entry["body"]); return
    L = log(); L.append(entry); LOG.write_text(json.dumps(L, indent=1))
    sc = entry["body"].get("score") if isinstance(entry["body"], dict) else None
    if isinstance(sc, (int, float)) and sc < BUG_FLOOR:
        alert(f"{name}: public score {sc} < {BUG_FLOOR} (ID {entry.get('id')}) -> auto-sending stopped, look for a bug")
    print("SENT", {k: v for k, v in entry.items() if k != "body"}, "body:", entry["body"])


def auto(names):
    C = {n: check(n) for n in dict.fromkeys(names)}
    C = {n: c for n, c in C.items() if c.get("nested_cv") is not None and c.get("md5_ok") and c.get("not_sent") and c.get("better")}
    if not C:
        print("NOTHING to send among", " ".join(names)); return
    n = max(C, key=lambda k: C[k]["nested_cv"]); c = C[n]
    if not c["checker"]:
        alert(f"{n}: checker FAILED on a candidate that satisfies the sending rule -> auto-sending stopped"); return
    if not c["sam_ok"]:
        print(f"HOLD_SAM {n} nested {c['nested_cv']:.6f}: uses {c['sam_runs']} SAM-derived runs, waiting for work/SAM_CLEARED"); return
    if not c["cadence_ok"]:
        print(f"WAIT {n} nested {c['nested_cv']:.6f}: cadence {c['wait_s']}s"); return
    send(n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--status", action="store_true"); ap.add_argument("--send")
    ap.add_argument("--cands", nargs="*", default=None)
    ap.add_argument("--auto", nargs="*", default=None)      # candidates of ONE loop pass: send the best clean one that satisfies the rule
    a = ap.parse_args()
    if a.auto is not None:
        auto(a.auto); return
    if a.send:
        send(a.send); return
    names = a.cands or sorted(p.name for p in (ROOT / "work/archive").iterdir())
    sent = {e["name"] for e in log()}
    for n in names:
        if n in sent:
            continue
        print(n, check(n))


if __name__ == "__main__":
    main()
