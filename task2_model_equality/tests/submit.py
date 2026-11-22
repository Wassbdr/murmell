"""usage: submit.py NAME 'ep-a+ep-b;ep-c+ep-d'  (groups; everything else singleton) [--at HH:MM:SS]"""
import os
import sys, json, time, pathlib, urllib.request, uuid, datetime
F = {'QA': '01,02,04,12,14,17,18,23,24,25,33', 'QB': '03,06,07,08,20,21,22,26,28,30,32', 'LL': '05,09,10,11,13,15,16,19,27,29,31'}
fam = {'ep-' + n: f for f, s in F.items() for n in s.split(',')}
name = sys.argv[1]; groups = [g.split('+') for g in sys.argv[2].split(';') if g]
lab = {e: f'{fam[e]}-{e[3:]}' for e in fam}
for i, g in enumerate(groups):
    assert len({fam[e] for e in g}) == 1, g
    for e in g: lab[e] = f'{fam[e]}-G{i}'
famlab = dict(fam)
if '--probe' in sys.argv:   # probe mode: family term forced to 0 (all groups share one fake family, singletons in cross-family triples)
    ing = {e for g in groups for e in g}; famlab = {e: 'F1' for e in ing}
    sing = {f: [e for e in sorted(fam) if fam[e] == f and e not in ing] for f in F}
    for f in F:
        for i, e in enumerate(sing[f]): famlab[e] = f'Z{i}'
D = pathlib.Path(__file__).parent / 'subs'; csv = 'endpoint,family,model\n' + ''.join(f'{e},{famlab[e]},{lab[e]}\n' for e in sorted(fam))
(D / f'{name}.csv').write_text(csv)
if '--dry' in sys.argv: print(csv); sys.exit()
if '--at' in sys.argv:
    h, m, s = map(int, sys.argv[sys.argv.index('--at') + 1].split(':')); now = datetime.datetime.now()
    t = now.replace(hour=h, minute=m, second=s, microsecond=0); w = (t - now).total_seconds()
    if w > 0: time.sleep(w)
key = os.environ['HACKATHON_API_KEY']; b = uuid.uuid4().hex
body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}.csv"\r\nContent-Type: text/csv\r\n\r\n{csv}\r\n--{b}--\r\n').encode()
for attempt in range(4):
    req = urllib.request.Request(os.environ.get('HACKATHON_BASE', 'http://34.147.229.55') + '/submit/33-model-equality', data=body, headers={'X-API-Key': key, 'Content-Type': f'multipart/form-data; boundary={b}'})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: st, txt = r.status, r.read().decode()
    except urllib.error.HTTPError as e: st, txt = e.code, e.read().decode()
    line = f'{time.strftime("%H:%M:%S")} {name} HTTP {st} {txt[:400]}'; print(line, flush=True)
    with (D / 'log.txt').open('a') as f: f.write(json.dumps({'t': time.time(), 'name': name, 'groups': groups, 'status': st, 'body': txt[:400]}) + '\n')
    if st == 429 and 'Wait' in txt:
        import re; m = re.search(r'Wait (\d+) seconds', txt); time.sleep(int(m.group(1)) + 3 if m else 30); continue
    break
