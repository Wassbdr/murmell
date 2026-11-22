import re,glob,sys
for f in sorted(glob.glob(f'output/{sys.argv[1]}_*.out')):
    tg=cl=al=None
    for line in open(f):
        if line.startswith('targets:'): tg=re.search(r'clean CE ([\d.]+/[\d.]+)',line).group(1)
        m=re.search(r'\[clean\] CE sub ([\d.]+/[\d.]+).*pred=([\d.]+)\s+acc_vs_pseudo=([\d.]+)',line)
        if m: cl=m.groups()
        m=re.search(r'\[alt  \] CE sub ([\d.]+/[\d.]+).*acc_vs_pseudo=([\d.]+)',line)
        if m: al=m.groups()
        m=re.search(r'PROXY subs/(\S+)\.pt:.*anchor KL ([\d.]+)',line)
        if m: print(f"{m.group(1):14} target {tg or '-':11} clean {cl[0]:11} acc {cl[2]} predfit {cl[1]} | alt {al[0]:11} acc {al[1]} | anchorKL {m.group(2)}")
        if 'FAILED' in line or 'SKIP' in line or 'weight-noise' in line: print('   ', line.strip()[:150])
