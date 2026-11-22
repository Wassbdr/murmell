"""Short candidate list for a fast last pass (Sun 11:22): the finals of pass P15 and the selector sources of P16 for every cell,
plus every complete SAM-chain run of TrustMark and StegaStamp built on the object-trained models (ensembles DS5 / DS6 / DS8 /
DS9 families and their snap variants, RS* runs and their snap variants).  Prints the names on one line."""
import json, os, re
R = "work/runs"; out = set()
for v in json.load(open("work/nested/P15.json"))["final"].values():
    out |= set(v[0])
for v in json.load(open("work/nested/P16.json")).get("sources", {}).values():
    if isinstance(v, dict):
        out |= set(v.get("all", []))
ok = lambda n: all(os.path.exists(f"{R}/{n}/{f}") and os.path.getsize(f"{R}/{n}/{f}") > 0 for f in ("train.npy", "test.npy", "args.json"))
for n in sorted(os.listdir(R)):
    if re.search(r"_xf\d$", n) or not ok(n) or not re.match(r"(DS[5689]_tm_|S[689]_ss_|S5e_ss_|RS\d?_|FTt_RS|FTxt_RS)", n):
        continue
    if json.load(open(f"{R}/{n}/args.json")).get("method") not in ("trustmark", "stegastamp"):
        continue
    if not n.endswith("_tta") and ok(n + "_tta"):
        continue
    out.add(n)
print(" ".join(sorted(r for r in out if r == "snapDF" or ok(r))))
