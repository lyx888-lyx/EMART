#!/usr/bin/env python3
"""Train auditable EMART V11 arms using the original trainer with DEV selection only.

Run from experiment/ (as the original finetune.py). No TEST data loader,
TEST inference, TEST labels in losses, or use of historical TEST scores.
Only prepared manifests produced by v11_audit_and_prepare.py are accepted.
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

import finetune as original
from dataloader.dataloader import (
    include_for_finetune, map_label, set_finetune_dataloader
)
from model.wavlm_plus import WavLMWrapper
from model.custom_roberta import RobertaCrossAttn
from model.prediction import TextAudioClassifierForCrossModalAttn

ROOT = Path(__file__).resolve().parents[1]


def load_split(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"train","dev"}:
        raise ValueError("Only V11 TRAIN/DEV-only manifests are allowed")
    splits = {}
    for name in ("train","dev"):
        out = []
        for entry in raw[name]:
            if not include_for_finetune(entry,"iemocap6"):
                raise ValueError(f"Unexpected label in {name}")
            entry = list(entry)
            entry[-1] = map_label(entry,"iemocap6")
            out.append(entry)
        if not out:
            raise ValueError(f"Empty split {name}")
        splits[name] = out
    return splits["train"], splits["dev"]


def scores(y, pred):
    return {"weighted_f1": float(100*f1_score(y,pred,average="weighted",zero_division=0)),
            "macro_f1": float(100*f1_score(y,pred,labels=list(range(6)),
                                           average="macro",zero_division=0)),
            "accuracy": float(100*accuracy_score(y,pred)),
            "uar": float(100*recall_score(y,pred,labels=list(range(6)),
                                         average="macro",zero_division=0))}


@torch.no_grad()
def dev_predictions(model, loader, device):
    y,pred=[],[]
    model.eval()
    for x,text,_,label,length in loader:
        logits,_,_,_=model(audio_input=x.to(device),text_input=text,length=length.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite DEV logits")
        y.extend(label.cpu().tolist())
        pred.extend(logits.argmax(-1).cpu().tolist())
    return y,pred


def native_metrics(d):
    return {k:float(d[k]) for k in ("mf1","uar","acc","loss")}


def train(args, arm, prepared, smoke):
    audit=json.loads((prepared/"audit.json").read_text())
    if arm not in audit["training_arms"]:
        raise ValueError(f"{arm} not authorized by actual cache match audit")
    train_rows,dev_rows=load_split(prepared/arm/"iemocap6_fold1.json")
    if len(train_rows)!=audit["splits"]["train"]["utterances"] or len(dev_rows)!=audit["splits"]["dev"]["utterances"]:
        raise ValueError("Manifest support differs from audit")
    args.split_dir=str(prepared/arm)
    # Same settings and original args object for all arms.
    if (args.dataset,args.audio_model,args.text_model,args.modal,args.speaker_mode,
        args.pooling_mode,args.multimodal_pooling,args.max_txt_len,args.load_pt,
        args.best_metric)!=( "iemocap6","wavlm","roberta-base","multimodal",
                            "self_other","curr_only","cls",256,True,"mf1"):
        raise ValueError("Use the fixed V11 launcher settings")
    if args.seed!=8 or args.batch_size!=64 or args.num_epochs!=(1 if smoke else 20):
        raise ValueError("V11 fixes seed 8, batch64 and epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("Original EMART text model uses CUDA. GPU required.")
    device=torch.device("cuda")
    root=ROOT/"results/v11_emart"/("smoke" if smoke else "full")/arm
    root.mkdir(parents=True,exist_ok=False)
    original.args=args
    original.fold_idx=1
    original.set_seed(args.seed)
    train_loader=set_finetune_dataloader(args,train_rows,is_train=True)
    dev_loader=set_finetune_dataloader(args,dev_rows,is_train=False)
    counts=torch.bincount(torch.tensor([r[-1] for r in train_rows]),minlength=6).float()
    if (counts==0).any():
        raise ValueError("Missing train emotion class")
    weights=counts.sum()/counts
    if args.class_weight_log:
        weights=weights.log()
    elif args.class_weight_norm:
        weights=weights/weights.sum()
    weights=weights.to(device)
    audio=WavLMWrapper(args).to(device)
    text=RobertaCrossAttn(args,audio).to(device)
    model=TextAudioClassifierForCrossModalAttn(
        audio_model=audio,text_model=text,audio_dim=768,text_dim=768,
        hidden_dim=args.hidden_dim,num_classes=6,dropout_prob=args.dr,
        cross_modal_atten=args.cross_modal_atten,modal=args.modal,
        multimodal_pooling=args.multimodal_pooling).to(device)
    # The original launcher intentionally retained these values for
    # faithful matched reproduction; min_lr may exceed initial LR.
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                         lr=args.learning_rate,weight_decay=args.weight_decay,
                         betas=(.9,.98))
    if args.warmup:
        raise ValueError("V11 uses original non-warmup scheduler")
    scheduler=ReduceLROnPlateau(opt,mode="min",patience=args.n_patience,
                               factor=.5,min_lr=args.min_lr)
    original.scheduler=scheduler
    scaler=GradScaler(enabled=True)
    best=-float("inf")
    history=[]
    checkpoint=root/"dev_best.pt"
    for e in range(args.num_epochs):
        original.epoch=e
        use_emo=bool(args.plutchik_instance_match and
                     (not args.plutchik_ipc_grad_start_ep or e>=args.plutchik_ipc_grad_start_ep-1))
        tr=original.train_epoch(train_loader,model,device,opt,weights,
                                args.at_barlow_align,use_emo,wandb_store=False,
                                scaler=scaler,scheduler=scheduler)
        va=original.validate_epoch(dev_loader,model,device,weights,
                                   args.at_barlow_align,use_emo,wandb_store=False)
        tr,va=native_metrics(tr),native_metrics(va)
        history.append({"epoch":e+1,"train":tr,"dev":va})
        if va["mf1"]>best:
            best=va["mf1"]
            torch.save({"model_state_dict":model.state_dict(),"epoch":e+1,
                        "dev_weighted_f1":best,"arm":arm},checkpoint)
        print(f"V11 {arm} epoch={e+1:02d} train_WF1={tr['mf1']:.2f} "
              f"dev_WF1={va['mf1']:.2f} best_dev={best:.2f}",flush=True)
        (root/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model.load_state_dict(saved["model_state_dict"],strict=True)
    y,pred=dev_predictions(model,dev_loader,device)
    if y!=[r[-1] for r in dev_rows]:
        raise RuntimeError("DEV prediction order mismatch")
    metrics=scores(y,pred)
    if abs(metrics["weighted_f1"]-saved["dev_weighted_f1"])>.025:
        raise RuntimeError("DEV checkpoint re-inference mismatch")
    result={"arm":arm,"seed":args.seed,"epochs":args.num_epochs,
            "selected_epoch":saved["epoch"],"selection":"dev_weighted_f1",
            "dev":metrics,"n":len(y),"labels":y,"predictions":pred,
            "utterance_ids":[r[0] for r in dev_rows],
            "checkpoint":str(checkpoint),"official_test_scored":False}
    (root/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("V11 DEV RESULT",json.dumps({k:v for k,v in result.items()
           if k not in ("labels","predictions","utterance_ids")}),flush=True)


def main():
    # Consume V11 options separately; keep the original EMART argument parser.
    import sys
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--v11-arm",required=True,choices=("legacy","aligned","role_fixed"))
    ap.add_argument("--v11-prepared",type=Path,default=ROOT/"results/v11_roles")
    ap.add_argument("--v11-smoke",action="store_true")
    v11,remaining=ap.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    args=original.parse_finetune_args()
    train(args,v11.v11_arm,v11.v11_prepared.resolve(),v11.v11_smoke)


if __name__=="__main__":
    main()
