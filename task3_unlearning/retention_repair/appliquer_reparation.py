"""Réparation de rétention instantanée : modele_repare = modele + a * delta (delta = réparation apprise sur lina_I_E4d05_tr03).
Vérifié : sur lina_I_E4d08, a=1 donne la même chose que l entraînement complet (rétention estimée 0,766 vs 0,770, KL oubli 0,003).
    python appliquer_reparation.py MODELE.pt SORTIE.pt [a=1.0]"""
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=True); d = torch.load("claude_delta_reparation.pt", map_location="cpu", weights_only=True)
a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
out = {k: (m[k].float() + a * d[k]).contiguous() for k in m}
assert all(torch.isfinite(v).all() for v in out.values()); torch.save(out, sys.argv[2]); print("ecrit", sys.argv[2], "a =", a)
