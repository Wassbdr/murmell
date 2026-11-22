"""Timed send (Sun 07:32): the 10-min server cooldown is shared with the teammate, whose automation takes each slot ~2 s after
it opens.  Same conditions as src.autosubmit.send (candidate READY: clean, rule, md5, checker, SAM clearance), but the checks
are done BEFORE the slot opens and the upload is fired at a given clock time, retried a few times on HTTP 429 only.
    python -m src.fastsend --name T01 --at 07:39:49 [--tries 6 --gap 0.7]
Logs exactly like autosubmit (work/submit_log.json); never prints credentials.
"""
import argparse
import json
import time
from datetime import datetime

from src import autosubmit as A
from src.data import ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--at", required=True)
    ap.add_argument("--tries", type=int, default=6); ap.add_argument("--gap", type=float, default=0.7)
    a = ap.parse_args()
    import requests
    c = A.check(a.name)
    if not all(c.get(k) for k in ("md5_ok", "not_sent", "better", "checker", "sam_ok")):
        print("NOT READY", c); return
    key, base = A.env()
    if not key or not base:
        print("credentials missing"); return
    spec = json.loads((ROOT / "work/archive" / a.name / "spec.json").read_text()); f = ROOT / spec["submission_file"]; blob = f.read_bytes()
    t_at = datetime.strptime(datetime.now().strftime("%Y-%m-%d ") + a.at, "%Y-%m-%d %H:%M:%S").timestamp()
    while time.time() < t_at:
        time.sleep(min(0.05, max(0.0, t_at - time.time())))
    for k in range(a.tries):
        t0 = time.time(); entry = {"name": a.name, "file": spec["submission_file"], "md5": spec["submission_md5"], "time": t0,
                                   "time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            r = requests.post(f"{base}/submit/{A.TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (f.name, blob, "application/octet-stream")}, timeout=(10, 180))
            try:
                body = r.json()
            except Exception:
                body = {"raw_text": r.text[:300]}
            entry.update(status=r.status_code, body=body, id=body.get("submission_id") if isinstance(body, dict) else None)
        except requests.exceptions.RequestException as e:
            entry.update(status="ERROR", body=repr(e)[:200], id=None)
        if entry.get("status") == 429:
            print(f"try {k + 1} {entry['time_str']} RATE-LIMITED: {entry['body']}", flush=True); time.sleep(a.gap); continue
        L = A.log(); L.append(entry); A.LOG.write_text(json.dumps(L, indent=1))
        sc = entry["body"].get("score") if isinstance(entry["body"], dict) else None
        if isinstance(sc, (int, float)) and sc < A.BUG_FLOOR:
            A.alert(f"{a.name}: public score {sc} < {A.BUG_FLOOR} (ID {entry.get('id')}) -> auto-sending stopped, look for a bug")
        print("SENT", {k_: v for k_, v in entry.items() if k_ != "body"}, "body:", entry["body"]); return
    print("NOT SENT: slot taken (429 on every try)")


if __name__ == "__main__":
    main()
