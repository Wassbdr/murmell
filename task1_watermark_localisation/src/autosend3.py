"""Login-side sender, version 3 (Sun 07:45).  Why: (1) the loop rewrites its candidates (M<HH>, S<HH>) in place at every pass,
so a candidate must be FROZEN under its own name before it is sent; (2) the 10-min server cooldown is shared with the
teammate, whose automation takes each slot ~2 s after it opens, so the upload must be fired at the opening second.
Rule (human 22 Nov 06:55, in src.autosubmit.check): clean candidate, nested CV > S11 (or within 0.002 and clearly different),
md5 never resent, checker VALID, SAM cleared.  Added here: a new candidate takes a slot only if its nested CV exceeds the
best nested CV we already sent by MIN_GAIN (the slot is shared; near-identical files bring no information).
Bug alarm (src.autosubmit.alert): public < 0.775 or checker failure -> work/STOP_AUTOSEND + work/ALERT.txt.
    nohup python -m src.autosend3 &          # log: work/autosend.log ; stop: touch work/STOP_AUTOSEND ; ends 10:25
"""
import hashlib
import json
import re
import time
from datetime import datetime

from src import autosubmit as A
from src.data import ROOT

LOOPLOG = ROOT / "output/atelier/0000_NIGHTLOOP.log"
OUT = ROOT / "work/autosend.log"
MIN_GAIN = 0.001
T_END = datetime(2026, 11, 22, 10, 25).timestamp()


def say(msg):
    with open(OUT, "a") as f:
        f.write(f"{datetime.now():%H:%M:%S} [autosend3] {msg}\n")


def last_pass():
    """(END line, candidate names) of the last COMPLETED pass, or (None, []) while a pass is running"""
    lines = LOOPLOG.read_text(errors="ignore").splitlines()
    if not lines or not lines[-1].startswith("=== END"):
        return None, []
    names = []
    for l in reversed(lines[:-1]):
        if l.startswith("=== PASS"):
            break
        m = re.match(r"NIGHT (\S+): nested", l)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return lines[-1], names


def free_name():
    for k in range(4, 41):
        n = f"T{k:02d}"; sp = ROOT / "work/archive" / n / "spec.json"
        if sp.exists() and sp.stat().st_size == 0:
            return n
    return None


def freeze(name):
    """copy candidate <name> (and its base, for a selector candidate) to free T-names; returns the frozen name"""
    d = json.loads((ROOT / "work/nested" / f"{name}.json").read_text()); extra = {"snapshot_of": f"{name} ({datetime.now():%H:%M})"}
    b = d.get("base")
    if b and not str(b).startswith("work/"):
        extra["base"] = freeze(b)
    dst = free_name()
    if dst is None:
        raise RuntimeError("no free T-name")
    sp = json.loads((ROOT / "work/archive" / name / "spec.json").read_text()); blob = (ROOT / sp["submission_file"]).read_bytes()
    if hashlib.md5(blob).hexdigest() != sp["submission_md5"]:
        raise RuntimeError(f"{name}: md5 mismatch (being rewritten?)")
    with open(ROOT / "work/archive" / dst / f"{dst}.pt", "wb") as f:
        f.write(blob)
    with open(ROOT / "work/nested" / f"{dst}.json", "w") as f:
        f.write(json.dumps(dict(d, **extra), indent=1))
    for a_, b_ in ((f"work/nested/{name}.npy", f"work/nested/{dst}.npy"), (f"work/test/{name}_bin.npy", f"work/test/{dst}_bin.npy")):
        with open(ROOT / b_, "wb") as f:
            f.write((ROOT / a_).read_bytes())
    with open(ROOT / "work/archive" / dst / "spec.json", "w") as f:      # written last: marks the T-name as used
        f.write(json.dumps(dict(sp, name=dst, submission_file=f"work/archive/{dst}/{dst}.pt", **extra), indent=1))
    return dst


def best_sent():
    return max([A.nested_cv(e["name"]) for e in A.log() if A.nested_cv(e["name"]) is not None and A.is_clean(e["name"])], default=0.0)


def post(name, key, base, spec, blob):
    import requests
    t0 = time.time(); entry = {"name": name, "file": spec["submission_file"], "md5": spec["submission_md5"], "time": t0,
                               "time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        r = requests.post(f"{base}/submit/{A.TASK_ID}", headers={"X-API-Key": key},
                          files={"file": (f"{name}.pt", blob, "application/octet-stream")}, timeout=(10, 180))
        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text[:300]}
        entry.update(status=r.status_code, body=body, id=body.get("submission_id") if isinstance(body, dict) else None)
    except requests.exceptions.RequestException as e:
        entry.update(status="ERROR", body=repr(e)[:200], id=None)
    return entry


def record(entry):
    L = A.log(); L.append(entry); A.LOG.write_text(json.dumps(L, indent=1))
    n = entry["name"]; d = json.loads((ROOT / "work/nested" / f"{n}.json").read_text()); b = entry["body"] if isinstance(entry["body"], dict) else {}
    with open(ROOT / "SUBMISSIONS.md", "a") as f:
        f.write(f"| {entry['time_str'][11:]} | {entry['file']} | {entry['md5']} | {d['cv_foldmean']:.6f} | **{b.get('score')}** | **{entry.get('id')}** | "
                f"{n} = {d.get('snapshot_of', '-')}, kind {d.get('kind')}, sent by src/autosend3.py (previous_best {b.get('previous_best')}) |\n")
    sc = b.get("score")
    say(f"SENT {n} nested {d['cv_foldmean']:.6f} public {sc} id {entry.get('id')} status {entry.get('status')}")
    if isinstance(sc, (int, float)) and sc < A.BUG_FLOOR:
        A.alert(f"{n}: public score {sc} < {A.BUG_FLOOR} (ID {entry.get('id')}) -> auto-sending stopped, look for a bug")


def wait_of(e):
    m = re.search(r"Wait (\d+) seconds", str(e.get("body")))
    return int(m.group(1)) if m else 60


def try_send(name):
    """one attempt. Returns (done, retry_at): done = sent or no longer sendable; retry_at = clock time of the next attempt"""
    c = A.check(name)
    if not all(c.get(k) for k in ("md5_ok", "not_sent", "better", "checker", "sam_ok")):
        say(f"DROP {name}: {c}"); return True, 0.0
    key, base = A.env()
    if not key or not base:
        say("credentials missing"); return True, 0.0
    spec = json.loads((ROOT / "work/archive" / name / "spec.json").read_text()); blob = (ROOT / spec["submission_file"]).read_bytes()
    e = post(name, key, base, spec, blob)
    if e.get("status") != 429:
        record(e); return True, 0.0
    wait = wait_of(e)
    if wait > 30:                                        # come back 25 s before the opening and ask the server again
        say(f"429 for {name}: slot opens in {wait}s"); return False, time.time() + wait - 25
    t_open = time.time() + wait - 1.2                    # the server counts whole seconds: start slightly early, retry fast
    while time.time() < t_open:
        time.sleep(0.05)
    for _k in range(10):
        e = post(name, key, base, spec, blob)
        if e.get("status") != 429:
            record(e); return True, 0.0
        time.sleep(0.35)
    wait = wait_of(e); say(f"slot lost for {name} (taken by the other sender); next opening in {wait}s")
    return False, time.time() + max(5, wait - 25)


def main():
    handled = None; pending = None; next_try = 0.0
    say("started")
    while time.time() < T_END and not A.STOP.exists():
        end, names = last_pass()
        if end and end != handled:
            handled = end; C = {}
            for n in names:
                c = A.check(n)
                if c.get("nested_cv") is not None and c.get("md5_ok") and c.get("not_sent") and c.get("better") and c.get("sam_ok"):
                    C[n] = c
            if C:
                n = max(C, key=lambda k: C[k]["nested_cv"]); cv = C[n]["nested_cv"]; ref = max(best_sent(), pending[1] if pending else 0.0)
                if not C[n]["checker"]:
                    A.alert(f"{n}: checker FAILED on a candidate that satisfies the sending rule -> auto-sending stopped"); break
                if cv > ref + MIN_GAIN:
                    try:
                        fz = freeze(n); pending = (fz, cv); say(f"[{end}] FROZEN {n} -> {fz} nested {cv:.6f} (best sent / pending {ref:.6f})")
                    except Exception as ex:                # e.g. the next pass started rewriting the files
                        say(f"[{end}] freeze of {n} failed: {ex!r}")
                else:
                    say(f"[{end}] {n} nested {cv:.6f} not above best sent / pending {ref:.6f} + {MIN_GAIN}: no send")
            else:
                say(f"[{end}] nothing sendable among {names}")
        if pending and time.time() >= next_try:
            done, next_try = try_send(pending[0])
            if done:
                pending = None; next_try = 0.0
        time.sleep(4)
    say("stopped (STOP file or 10:25)")


if __name__ == "__main__":
    main()
