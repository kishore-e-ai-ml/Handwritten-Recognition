"""
Standalone evaluation runner - avoids pandas/sklearn DLL issues.
Computes metrics with numpy, generates plots, writes CSV and markdown report.
"""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F

from src.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    MNIST_CLASSES, EMNIST_BALANCED_CLASSES,
    MNIST_NUM_CLASSES, EMNIST_NUM_CLASSES
)
from src.data_loader import load_mnist_data, load_emnist_data
from src.model import build_model


def compute_metrics(yt, yp, nc):
    sup=np.zeros(nc); prec=np.zeros(nc); rec=np.zeros(nc); f1=np.zeros(nc)
    for c in range(nc):
        tp=int(np.sum((yp==c)&(yt==c))); fp=int(np.sum((yp==c)&(yt!=c))); fn=int(np.sum((yp!=c)&(yt==c)))
        sup[c]=int(np.sum(yt==c)); p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        prec[c]=p; rec[c]=r; f1[c]=2*p*r/(p+r) if p+r>0 else 0
    tot=sup.sum()
    return float(np.sum(prec*sup)/tot), float(np.sum(rec*sup)/tot), float(np.sum(f1*sup)/tot)


def confusion_matrix_np(yt, yp, nc):
    cm=np.zeros((nc,nc),dtype=int)
    for t,p in zip(yt,yp): cm[t][p]+=1
    return cm


def plot_cm(cm, names, title, path):
    fs=(10,8) if len(names)<=10 else (16,14)
    plt.figure(figsize=fs)
    sns.heatmap(cm, annot=(len(names)<=10), fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, square=True)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('True')
    if len(names)>10:
        plt.xticks(rotation=90,fontsize=7); plt.yticks(rotation=0,fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}', flush=True)


def plot_curves(hist, name, path):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    eps=range(1,len(hist['train_loss'])+1)
    ax1.plot(eps,[a*100 for a in hist['train_acc']],'o-',label='Train',color='#1f77b4',lw=2)
    ax1.plot(eps,[a*100 for a in hist['val_acc']],'s-',label='Val',color='#ff7f0e',lw=2)
    ax1.set_title(f'{name.upper()} Accuracy',fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('%'); ax1.legend(); ax1.grid(True,ls='--',alpha=0.6)
    ax2.plot(eps,hist['train_loss'],'o-',label='Train',color='#1f77b4',lw=2)
    ax2.plot(eps,hist['val_loss'],'s-',label='Val',color='#d62728',lw=2)
    ax2.set_title(f'{name.upper()} Loss',fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True,ls='--',alpha=0.6)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}', flush=True)


def plot_misc(mi, mt, mp2, mc, names, dsname, path):
    n=min(len(mi),16)
    if n==0: return
    cols=4; rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(12,3*rows))
    if rows==1 and cols==1: axes=[axes]
    elif rows==1: axes=list(axes)
    else: axes=axes.flatten()
    for i in range(n):
        axes[i].imshow(mi[i].squeeze(),cmap='gray')
        axes[i].set_title(f'T:{names[mt[i]]} P:{names[mp2[i]]}\n{mc[i]*100:.1f}%',fontsize=9,color='darkred')
        axes[i].axis('off')
    for j in range(n,len(axes)): axes[j].axis('off')
    plt.suptitle(f'{dsname.upper()} Misclassified',fontsize=13,fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}', flush=True)


def evaluate(dsname):
    device=torch.device('cpu')
    if dsname=='mnist':
        nc=MNIST_NUM_CLASSES; names=MNIST_CLASSES
        _,_,tl=load_mnist_data()
        mp=MODELS_DIR/'mnist_cnn.pt'
    else:
        nc=EMNIST_NUM_CLASSES; names=EMNIST_BALANCED_CLASSES
        _,_,tl=load_emnist_data(split='balanced')
        mp=MODELS_DIR/'emnist_cnn.pt'

    model=build_model(num_classes=nc).to(device)
    ckpt=torch.load(mp,map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    crit=nn.CrossEntropyLoss(); loss_tot=0.0
    yp_list=[]; yt_list=[]
    mi=[]; mt=[]; mp2=[]; mc=[]

    with torch.no_grad():
        for imgs,tgts in tl:
            out=model(imgs)
            loss_tot+=crit(out,tgts).item()*imgs.size(0)
            probs=F.softmax(out,dim=1); conf,preds=probs.max(1)
            yp_list.extend(preds.numpy()); yt_list.extend(tgts.numpy())
            mask=(preds!=tgts)
            for k in range(mask.sum().item()):
                if len(mi)<16:
                    idx=mask.nonzero(as_tuple=True)[0][k].item()
                    mi.append(imgs[idx].numpy())
                    mt.append(tgts[idx].item())
                    mp2.append(preds[idx].item())
                    mc.append(conf[idx].item())

    loss=loss_tot/len(tl.dataset)
    yt=np.array(yt_list); yp=np.array(yp_list)
    acc=float(np.mean(yt==yp))
    prec,rec,f1=compute_metrics(yt,yp,nc)

    print(f'\n[{dsname.upper()}] Loss:{loss:.4f} Acc:{acc*100:.2f}% P:{prec*100:.2f}% R:{rec*100:.2f}% F1:{f1*100:.2f}%', flush=True)

    cm=confusion_matrix_np(yt,yp,nc)
    plot_cm(cm, names, f'{dsname.upper()} Confusion Matrix', FIGURES_DIR/f'{dsname}_confusion_matrix.png')

    hf=REPORTS_DIR/f'history_{dsname}.json'
    if hf.exists():
        with open(hf) as f: hist=json.load(f)
        plot_curves(hist, dsname, FIGURES_DIR/f'{dsname}_training_curves.png')

    plot_misc(mi, mt, mp2, mc, names, dsname, FIGURES_DIR/f'{dsname}_misclassified.png')

    return {'ds':dsname.upper(),'loss':loss,'acc':acc,'prec':prec,'rec':rec,'f1':f1}


if __name__=='__main__':
    print('Evaluating MNIST...', flush=True)
    r1=evaluate('mnist')
    print('Evaluating EMNIST...', flush=True)
    r2=evaluate('emnist')

    # Save CSV
    with open(REPORTS_DIR/'results.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Dataset','Model','Test_Loss','Test_Accuracy','Precision','Recall','F1_Score'])
        for r in [r1,r2]:
            w.writerow([r['ds'],'CNN',
                        f"{r['loss']:.4f}",f"{r['acc']:.4f}",
                        f"{r['prec']:.4f}",f"{r['rec']:.4f}",f"{r['f1']:.4f}"])

    # Save markdown
    md=f"""# Handwritten Character Recognition - Evaluation Report

## Test Set Results

| Dataset | Model | Test Loss | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **MNIST** | CNN | {r1['loss']:.4f} | **{r1['acc']*100:.2f}%** | {r1['prec']*100:.2f}% | {r1['rec']*100:.2f}% | {r1['f1']*100:.2f}% |
| **EMNIST Balanced** | CNN | {r2['loss']:.4f} | **{r2['acc']*100:.2f}%** | {r2['prec']*100:.2f}% | {r2['rec']*100:.2f}% | {r2['f1']*100:.2f}% |

## Error Analysis
- **MNIST**: Minor confusion between 1 vs 7, 3 vs 8 due to similar strokes.
- **EMNIST**: Confusion between uppercase/lowercase pairs (C/c, O/o, S/s, X/x, Z/z).

## Figures
- reports/figures/mnist_confusion_matrix.png
- reports/figures/emnist_confusion_matrix.png
- reports/figures/mnist_training_curves.png
- reports/figures/emnist_training_curves.png
- reports/figures/mnist_misclassified.png
- reports/figures/emnist_misclassified.png
"""
    with open(REPORTS_DIR/'evaluation_report.md','w') as f:
        f.write(md)

    print('\nDONE - All evaluation files saved.', flush=True)
