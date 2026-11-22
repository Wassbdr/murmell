"""Train a successor from official results on the existing GPU and nominate validated weights."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
from statistics import median
import subprocess
import sys
import time

from piloter_soumissions import QUEUE, append, receipts, rows
from submit_when_available import ROOT, write_json


JOBS = ROOT/"iterations_recus.json"
EXTANT = ROOT/"aws-jobs.jsonl"
CONFIG = json.loads((ROOT/"aws-instance.json").read_text())
S3 = "s3://"+CONFIG["bucket"]+"/"+CONFIG["prefix"]
AWS = ["aws","--no-cli-pager","--profile",CONFIG["profile"],"--region",CONFIG["region"]]


def aws(*args):
    return subprocess.run([*AWS,*args],check=True,capture_output=True,text=True,timeout=60).stdout.strip()


def fetch(key,path):
    tmp = path.with_name(path.name+".transfer")
    aws("s3","cp",S3+"/"+key,str(tmp),"--only-show-errors")
    tmp.replace(path)


def next_training_receipt(known, jobs):
    record = max(known.values(), key=lambda result: (result["score"], result["submission_id"]))
    training = jobs.get(str(record["submission_id"]))
    if training is None or (training["status"] == "failed" and not training.get("redispatch")):
        return record
    return max(known.values(), key=lambda result: result["submission_id"])


def plan(latest, known):
    parent = max(known.values(),key=lambda v:(v["score"], v["submission_id"]))
    positive = latest["score"] >= latest.get("previous_best",0)
    # Calibrated targets have produced gains. After a regression, reduce their
    # influence and sample preserved images according to measured output drift.
    return {"id":latest["submission_id"],"parent":parent["model"],
            "parent_score":parent["score"],"fraction":.65 if positive else .4,
            "retention_weight":23 if positive else 30,"anchors":"drift",
            "target_ensemble":"current" if positive else "include_parent",
            "status":"starting"}


def comparison_id(job):
    return str(job["id"])+("_reprise" if job.get("attempt")=="retry_alt" else "")


def start(job):
    num = job["id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.pt",job["parent"]):
        raise ValueError("Unexpected parent filename")
    # The initial run may already have been dispatched while its receipt was being reported.
    marker = f"Sujet 3 - ajuster la distillation au score officiel {num}"
    if job.get("attempt") in ("retry","retry_alt"):
        marker = f"Sujet 3 - reprendre la distillation après le reçu {num}"
    if job.get("redispatch"):
        marker += " (reprise du calcul)"
    previous = [r for r in rows(EXTANT) if r.get("name")==marker]
    if previous:
        job.update(command=previous[-1]["id"],status="remote")
        return
    file = "tools/codex_gold/entrainer_apres_recu.sh"
    args = [str(num),job["parent"],str(job["fraction"]),str(job["retention_weight"]),
            job["anchors"],job.get("target_ensemble","current"),job.get("attempt","initial")]
    commands = ("set -euo pipefail\ncd /workspace/unlearning\n"
                +shlex.join(["aws","--region",CONFIG["region"],"s3","cp",
                             S3+"/codex/calibrated_gold/entrainer_apres_recu.sh",file,"--only-show-errors"])
                +"\n"+shlex.join(["bash",file,*args]))
    remote = "sudo -u ubuntu -H bash -c "+shlex.quote(commands)
    parameters = json.dumps({"commands":[remote],"executionTimeout":["7200"]})
    command = aws("ssm","send-command","--instance-ids",CONFIG["id"],"--document-name",
                  "AWS-RunShellScript","--comment",marker,"--parameters",parameters,
                  "--query","Command.CommandId","--output","text")
    append(EXTANT,{"name":marker,"id":command})
    job.update(command=command,status="remote")
    print("ENTRAINEMENT",num,job["parent"],command,flush=True)


def lineage_projection(job, delta, known):
    ratios=[]
    for rec in sorted(known.values(),key=lambda value:value["submission_id"]):
        if not re.fullmatch(r"codex_prob_recu_\d+(?:_reprise)?\.pt",rec["model"]):
            continue
        path=ROOT/Path(rec["model"]).with_suffix(".json")
        if not path.is_file():
            continue
        metric=json.loads(path.read_text())
        parent=known.get(metric.get("source"))
        if parent is None or not metric.get("fresh_final"):
            continue
        published=rec["score"]-parent["score"]
        measured=metric["fresh_final"]["quality"]-metric["fresh_source"]["quality"]
        if measured>.001:
            ratios.append(published/measured)
    slope=median(ratios[-3:]) if len(ratios)>=3 else .65
    return job["parent_score"]+max(-.045,min(.045,delta*slope)),len(ratios)


def harvest(job, cache, known):
    num = job["id"]
    tag = "codex_prob_recu_"+comparison_id(job)
    report = ROOT/(tag+".json")
    fetch("aws-logs/"+tag+".json",report)
    metrics = json.loads(report.read_text())
    job["retry_reviewed"]=True
    no_weights=metrics.get("selected_epoch")==0
    regressed=(bool(metrics.get("fresh_final")) and
               metrics["fresh_final"]["quality"]<=metrics["fresh_source"]["quality"])
    if (no_weights or regressed) and job.get("attempt","initial")=="initial" and job.get("target_ensemble")!="include_parent":
        # Keep existing weights intact when the independent evaluation rejects
        # them. A smaller step also uses the parent's previously successful targets.
        job.update(status="starting",attempt="retry" if no_weights else "retry_alt",first_command=job.get("command"),
                   target_ensemble="include_parent",fraction=.4,retention_weight=30)
        for field in ("command","weights_ready","quick_checked"):
            job.pop(field,None)
        print("REPRISE_CIBLES_PARENT",num,job["parent"],flush=True)
        return
    if not metrics.get("selected_epoch") or "fresh_final" not in metrics:
        job["status"]="parent_preferred"
        return
    old, new = metrics["fresh_source"]["quality"], metrics["fresh_final"]["quality"]
    if new <= old:
        job["status"]="validation_regressed"
        return
    digest_file = cache/(tag+".sha256")
    fetch("aws-logs/"+tag+".sha256",digest_file)
    digest = digest_file.read_text().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Invalid weight digest")
    ranked = ROOT/(tag+"_classement.json")
    fetch("aws-logs/"+ranked.name,ranked)
    comparison = json.loads(ranked.read_text())
    predictions = comparison["models"]
    prediction = predictions[tag]
    extra = cache/("supplement_"+comparison_id(job)+".pt")
    fetch("caches/"+extra.name,extra)
    subprocess.run([sys.executable,str(ROOT/"recalibrate_receipts.py"),str(cache)],check=True,cwd=ROOT)
    last = max(v["submission_id"] for v in known.values())
    rank_reports=sorted(ROOT.glob("classement_apres_*.json"),
                        key=lambda f:int(f.stem.split("_")[-1]),reverse=True)
    common = None
    for path in rank_reports:
        score = json.loads(path.read_text())
        if tag in score["candidates"]:
            common=score["record_official"]+score["candidates"][tag]["median_change"]
            break
    champion = max(known.values(), key=lambda result: result["score"])
    reference = predictions.get(Path(champion["model"]).stem)
    lina = prediction["anchored"]
    if reference is not None:
        lina += champion["score"] - reference["anchored"]
    proxy,_ = lineage_projection(job,new-old,known)
    values=[proxy,lina]
    if common is not None:
        values.append(common)
    frozen_file=ROOT/(tag+"_fige.json")
    try:
        fetch("aws-logs/"+frozen_file.name,frozen_file)
    except subprocess.CalledProcessError:
        pass
    if frozen_file.is_file():
        frozen=json.loads(frozen_file.read_text())
        checks=frozen.get("historical",[])
        models = frozen.get("models",{})
        if len(checks)>=2 and tag in models:
            f = models[tag]["median"]
            recent = models.get(Path(champion["model"]).stem)
            if recent is not None:
                values.append(f + champion["score"] - recent["median"])
            elif frozen.get("parent") == Path(champion["model"]).stem:
                values.append(f)
    estimate = sum(values)/len(values)
    current = next((r for r in reversed(rows(QUEUE)) if r["sha256"]==digest),None)
    if current is None or current.get("complete_comparison")!=num:
        evidence=(f"Validation sur de nouvelles vues {old:.4f} à {new:.4f}. "
                  f"Estimation de Lina {lina:.4f} avec {prediction['configs']} réglages.")
        entry={"model":tag+".pt","sha256":digest,"family":"distillation-probabilites",
               "parent":job["parent"],"basis":last,"forecast":min(1.,max(0.,round(estimate,6))),
               "complete_comparison":num,"evidence":evidence}
        append(QUEUE,entry)
        print("CANDIDAT_VALIDE",tag,round(estimate,4),flush=True)
    job.update(status="nominated",sha256=digest,forecast=estimate)


def weights_ready(job, cache):
    if job.get("weights_ready") or job["status"] == "harvest":
        return True
    if job["status"] != "remote":
        return False
    name = "codex_prob_recu_"+comparison_id(job)+".sha256"
    if not (cache / name).is_file():
        try:
            aws("s3api","head-object","--bucket",CONFIG["bucket"],"--key",
                CONFIG["prefix"]+"/aws-logs/"+name,"--query","ContentLength","--output","text")
        except subprocess.CalledProcessError as error:
            if "404" in (error.stderr or "") or "Not Found" in (error.stderr or ""):
                return False
            raise
    job["weights_ready"] = True
    return True


def nominate_early(job, cache, known):
    if job.get("quick_checked") or not weights_ready(job, cache):
        return
    num = job["id"]
    tag = "codex_prob_recu_"+comparison_id(job)
    form = ROOT/(tag+".json")
    digest_file = cache/(tag+".sha256")
    for file in (form, digest_file):
        if not file.is_file():
            fetch("aws-logs/"+file.name, file)
    metric = json.loads(form.read_text())
    if not metric.get("fresh_final"):
        fetch("aws-logs/"+form.name, form)
        return
    old, new = metric["fresh_source"]["quality"], metric["fresh_final"]["quality"]
    digest = digest_file.read_text().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Invalid weight digest")
    current = next((r for r in reversed(rows(QUEUE)) if r["sha256"]==digest),None)
    if tag+".pt" in known or new <= old:
        job["quick_checked"] = True
        return
    latest = max(v["submission_id"] for v in known.values())
    estimate, count = lineage_projection(job,new-old,known)
    if not current and count>=3:
        append(QUEUE,{"model":tag+".pt","sha256":digest,"family":"distillation-probabilites",
              "parent":job["parent"],"basis":latest,"forecast":round(estimate,6),"lineage_only":True,
              "evidence":f"Poids vérifiés. Sur de nouvelles vues, la qualité passe de {old:.4f} à {new:.4f}. "
                         f"Le rapport médian des trois derniers gains de la même famille donne {estimate:.4f}."})
        print("CANDIDAT_LIGNEE",tag,round(estimate,4),flush=True)
    extra = cache/("supplement_"+comparison_id(job)+".pt")
    if not extra.is_file():
        try:
            fetch("caches/"+extra.name, extra)
        except subprocess.CalledProcessError as e:
            if "404" in (e.stderr or "") or "Not Found" in (e.stderr or ""):
                return
            raise
    if current and not current.get("lineage_only"):
        job["quick_checked"] = True
        return
    subprocess.run([sys.executable,str(ROOT/"recalibrate_receipts.py"),str(cache)],check=True,cwd=ROOT)
    file = ROOT/f"classement_apres_{latest}.json"
    if not file.is_file():
        return
    comparison = json.loads(file.read_text())
    score = comparison["candidates"].get(tag)
    if score is None:
        return
    if score["median_change"] > 0 or (current and current.get("lineage_only")):
        common = comparison["record_official"] + score["median_change"]
        forecast = (estimate+common)/2 if count>=3 else common
        append(QUEUE,{"model":tag+".pt","sha256":digest,"family":"distillation-probabilites",
               "parent":job["parent"],"basis":latest,"forecast":round(forecast,6),
               "evidence":f"Poids vérifiés. Validation sur de nouvelles vues de {old:.4f} à {new:.4f}, "
                          f"estimation sur les six vues communes {common:.4f}."})
        print("CANDIDAT_MESURE",tag,round(forecast,4),flush=True)
    job["quick_checked"] = True


def watch(cache):
    with (ROOT/".iterations_recus.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs = json.loads(JOBS.read_text()) if JOBS.is_file() else {}
        while json.loads((ROOT/"suivi.json").read_text()).get("boucle",{}).get("active"):
            known=receipts()
            latest=next_training_receipt(known,jobs)
            if max(v["score"] for v in known.values())>=1:
                return
            recent=jobs.get(str(latest["submission_id"]))
            if recent and recent["status"]=="failed" and not recent.get("redispatch"):
                recent.update(status="starting",redispatch=True,failed_command=recent.get("command"))
            if (recent and recent.get("attempt","initial")=="initial" and recent.get("target_ensemble")!="include_parent"
                    and (recent["status"]=="validation_regressed" or
                         (recent["status"]=="parent_preferred" and not recent.get("retry_reviewed")))):
                recent["status"]="harvest"
            ongoing = [job for job in jobs.values() if job["status"] in ("starting","remote","harvest")]
            try:
                # The shared node serialises GPU work. Once weights exist, validation
                # of an earlier model can finish while a new receipt is used for training.
                if str(latest["submission_id"]) not in jobs and all(weights_ready(j,cache) for j in ongoing):
                    job = plan(latest, known)
                    jobs[str(job["id"])] = job
                    ongoing.insert(0,job)
                    write_json(JOBS,jobs)
            except (OSError,subprocess.SubprocessError,ValueError) as exc:
                print("REPRISE_PLAN",latest["submission_id"],str(exc)[:230],flush=True)
            for active in ongoing:
                try:
                    if active["status"]=="starting":
                        start(active)
                    if active["status"]=="remote":
                        state=aws("ssm","get-command-invocation","--instance-id",CONFIG["id"],
                                  "--command-id",active["command"],"--query","Status","--output","text")
                        if state=="Success":
                            active["status"]="harvest"
                        elif state not in ("Pending","InProgress","Delayed","Cancelling"):
                            active.update(status="failed",remote_state=state)
                            print("ENTRAINEMENT_A_EXAMINER",active["id"],state,flush=True)
                        else:
                            nominate_early(active,cache,known)
                    if active["status"]=="harvest":
                        harvest(active,cache,known)
                    write_json(JOBS,jobs)
                except (OSError,subprocess.SubprocessError,KeyError,ValueError) as exc:
                    message=getattr(exc,"stderr","") or str(exc)
                    print("REPRISE_ITERATION",active["id"],message.strip()[:230],flush=True)
            time.sleep(18)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cache",type=Path)
    watch(parser.parse_args().cache)
