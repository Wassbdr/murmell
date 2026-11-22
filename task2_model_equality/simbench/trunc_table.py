"""trunc_table.py - reads fam/*/trunc.log (output of the lead's trunc_test.py) and prints observed / expected tail counts by kind.

    python trunc_table.py fam"""
import csv, os, re, sys, collections
root = sys.argv[1]; rows = collections.defaultdict(list)
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name, 'trunc.log')
    if not os.path.exists(p): continue
    truth = {r['endpoint'][-2:]: r for r in csv.DictReader(open(os.path.join(root, name, 'truth.csv')))}
    for l in open(p, errors='ignore'):
        m = re.match(r'^(\d\d): +(\d+) T ([0-9.]+) \| (.*) \| (.*) \| max ([0-9.]+)', l)
        if not m: continue
        ps = dict((k, (int(o), float(x))) for k, o, x in re.findall(r'(p[0-9.]+) +(\d+)/ *([0-9.]+)', m.group(4)))
        ks = dict((k, (int(o), float(x))) for k, o, x in re.findall(r'(k\d+) +(\d+)/ *([0-9.]+)', m.group(5)))
        rows[truth[m.group(1)]['kind']].append(dict(fam=name, T=float(m.group(3)), n=int(m.group(2)), mx=float(m.group(6)), **ps, **ks))
def korder(k): m = k.split('+')[0]; return (m != 'base', m, k)
print('| kind | family | tokens | fitted T | beyond rank 50: observed / expected | beyond rank 100 | beyond nucleus 0.95 | beyond nucleus 0.99 | largest mass before a sampled token |')
print('|---|---|---|---|---|---|---|---|---|')
for k in sorted(rows, key=korder):
    for r in rows[k]:
        print('| %s | %s | %d | %.2f | %d / %.1f | %d / %.1f | %d / %.1f | %d / %.1f | %.4f |' % (k, r['fam'], r['n'], r['T'], *r['k50'], *r['k100'], *r['p0.95'], *r['p0.99'], r['mx']))
