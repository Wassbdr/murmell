import sys, torch
name, alphas = sys.argv[1], [float(a) for a in sys.argv[2:]]
so = torch.load('../model.pt', map_location='cpu', weights_only=True)
su = torch.load(f'subs/{name}.pt', map_location='cpu', weights_only=True)
for a in alphas:
    sd = {k: (so[k].float() + a * (su[k].float() - so[k].float())) for k in so}
    assert all(torch.isfinite(v).all() for v in sd.values())
    torch.save(sd, f'subs/{name}_a{a}.pt'); print('saved', f'subs/{name}_a{a}.pt')
