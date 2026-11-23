"""Slot-by-slot sender (Sun 09:22, human's carte blanche of 09:13: best private score, use every slot).
work/sendqueue.txt = candidate names, best first, one per line ('#' = comment); the file is re-read until 8 s before each
slot opening, so a better candidate can be put on top at any time.  At each opening (last successful send + 10 min, shared
with the teammate) the first name that is clean, unsent (md5), VALID for the checker and SAM-cleared is uploaded at the
opening second (retries on HTTP 429 only).  Every send is logged like tools.autosubmit (work/submit_log.json, SUBMISSIONS.md).
Bug alarm unchanged (public < 0.775 -> work/STOP_AUTOSEND).  Stop: touch work/STOP_AUTOSEND.  Ends 11:52.
    nohup python -m tools.sendqueue &          # log: work/autosend.log
"""
import json
import time
from datetime import datetime

from tools import autosubmit as A
from tools.autosend3 import post, record, say, wait_of
from src.data import ROOT

Q = ROOT / "work/sendqueue.txt"
T_END = datetime(2026, 11, 22, 11, 52).timestamp()


def pick():
    if not Q.exists():
        return None
    for n in [l.strip() for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")]:
        sp = ROOT / "work/archive" / n / "spec.json"
        if not sp.exists() or sp.stat().st_size == 0:
            continue
        c = A.check(n)
        if c.get("nested_cv") is not None and all(c.get(k) for k in ("md5_ok", "not_sent", "checker", "sam_ok")) and A.is_clean(n):
            return n
    return None


def main():
    say("sendqueue started"); key, base = A.env(); told = None; foreign = 0.0     # foreign = opening time learnt from a 429 (send by someone else)
    while time.time() < T_END and not A.STOP.exists():
        L = A.log(); last = max([e["time"] for e in L if e.get("status") == 200], default=0.0); t_open = max(last + 600.0, foreign)
        n = pick()
        if n is None:
            time.sleep(5); continue
        if time.time() < t_open - 8:
            if told != (n, int(t_open)):
                say(f"next: {n} at {datetime.fromtimestamp(t_open):%H:%M:%S}"); told = (n, int(t_open))
            time.sleep(min(5.0, t_open - 8 - time.time())); continue
        n = pick()                                        # final read of the queue
        if n is None:
            continue
        spec = json.loads((ROOT / "work/archive" / n / "spec.json").read_text()); blob = (ROOT / spec["submission_file"]).read_bytes()
        if time.time() > t_open + 2:                       # stale opening time (someone else sent since): one probe, learn the real opening
            e = post(n, key, base, spec, blob)
            if e.get("status") == 429:
                w = wait_of(e); foreign = time.time() + w - 0.5
                if told != ("probe", int(foreign)):
                    say(f"cooldown running (other sender): opening at about {datetime.fromtimestamp(foreign):%H:%M:%S}"); told = ("probe", int(foreign))
                time.sleep(1.0 if w > 9 else 0.0)
                if w > 9:
                    continue
                t_open = foreign
            else:
                record(e); continue
        # burst: staggered parallel uploads around the opening (another sender of the team fires at the same second);
        # only the first request the server sees after the opening can succeed, the others get HTTP 429 and are dropped
        import threading
        res = []; lock = threading.Lock()

        def shot(delay):
            t_fire = t_open + delay
            while time.time() < t_fire:
                time.sleep(0.005)
            r = post(n, key, base, spec, blob)
            with lock:
                res.append(r)

        th = [threading.Thread(target=shot, args=(d,), daemon=True) for d in (-1.6, -1.3, -1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.8, 1.2)]
        for t_ in th:
            t_.start()
        for t_ in th:
            t_.join(timeout=200)
        ok = [r for r in res if r.get("status") != 429]
        e = ok[0] if ok else (res[-1] if res else None)
        if e is not None and e.get("status") == 429:
            w = max(wait_of(r) for r in res)
            if w > 20:
                foreign = time.time() + w - 0.5; say(f"slot taken by another sender; next opening at about {datetime.fromtimestamp(foreign):%H:%M:%S}"); time.sleep(2.0)
            else:                                         # opening later than expected: finish with single shots
                time.sleep(max(0.0, w - 1.0))
                for _ in range(20):
                    e = post(n, key, base, spec, blob)
                    if e.get("status") != 429 or wait_of(e) > 20:
                        break
                    time.sleep(0.1)
        if e is not None and e.get("status") != 429:
            record(e)
            if e.get("status") != 200:                    # server error: do not loop on it
                say(f"non-200 answer for {n}: {e.get('status')} {str(e.get('body'))[:120]}"); time.sleep(20)
    say("sendqueue stopped (STOP file or 11:52)")


if __name__ == "__main__":
    main()
