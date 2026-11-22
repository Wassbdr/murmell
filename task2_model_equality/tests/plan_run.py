"""Generic direct collector. usage: plan_run.py PLAN.jsonl OUT.jsonl   PLAN lines: {"endpoint","id","prompt","max_tokens"}; resumable on (endpoint,id)."""
import os
import json, sys, time, threading, pathlib, urllib.request, urllib.error, collections
KEY = os.environ['HACKATHON_API_KEY']; BASE = os.environ.get('HACKATHON_BASE', 'http://34.147.229.55') + '/33-model-equality/v1'
plan = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; out_p = sys.argv[2]
done = set()
if pathlib.Path(out_p).exists():
    for l in open(out_p): r = json.loads(l); done.add((r['endpoint'], r['id']))
jobs = collections.deque(j for j in plan if (j['endpoint'], j['id']) not in done)
lock = threading.Lock(); out = open(out_p, 'a'); G = dict(ok=0, fail=0); busy = set()
def call(body):
    req = urllib.request.Request(BASE + '/chat/completions', data=json.dumps(body).encode(), headers={'X-API-Key': KEY, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:300]
    except Exception as e: return -1, repr(e)[:200]
def worker():
    while G['fail'] < 6:
        with lock:
            job = None
            for i, j in enumerate(jobs):
                if j['endpoint'] not in busy: job = j; del jobs[i]; busy.add(j['endpoint']); break
            if job is None and not jobs: return
        if job is None: time.sleep(1); continue
        t = time.time(); s, b = call({'model': job['endpoint'], 'max_tokens': job['max_tokens'], 'messages': [{'role': 'user', 'content': job['prompt']}]})
        with lock:
            busy.discard(job['endpoint'])
            if s == 200:
                G['ok'] += 1; c = b['choices'][0]
                out.write(json.dumps(dict(job, response=c['message']['content'], finish_reason=c.get('finish_reason'), budget=b.get('budget'), latency_s=round(time.time() - t, 2), at=time.time()), ensure_ascii=False) + '\n'); out.flush()
            elif s == 429: jobs.appendleft(job)
            else: G['fail'] += 1; print('FAIL', job['endpoint'], job['id'], s, str(b)[:150], flush=True)
        if s == 429: time.sleep(3)
th = [threading.Thread(target=worker) for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2)]; [x.start() for x in th]; [x.join() for x in th]
print('finished ok', G['ok'], 'fail', G['fail'], flush=True)
