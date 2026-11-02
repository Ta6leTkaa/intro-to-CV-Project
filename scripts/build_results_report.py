"""Generate Stage 5 LaTeX tables and annotated failures from measured test outputs.

Run after evaluate_test.py; compile planning/latex/results_checkpoint.tex with pdflatex
(twice) or tectonic from planning/latex. This script does not run/tune any models.
"""
import csv
import json
import os
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from detection_metrics import box_iou, match_detections
from evaluate_models import ROOT, write_csv

OUT = ROOT / 'reports/results_checkpoint'
TEX = ROOT / 'planning/latex'
FIG = OUT / 'figures'


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def esc(s):
    return str(s).replace('_',r'\_')


def pm(values, digits=3):
    return f'{statistics.mean(values):.{digits}f} $\\pm$ {statistics.stdev(values):.{digits}f}'


def table(name, headers, rows, spec=None):
    spec = spec or 'l'+'r'*(len(headers)-1)
    content = '\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    content += '\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    content += '\n\\bottomrule\n\\end{tabular}\n'
    (TEX / name).write_text(content)


def comparison_figures(names, classes, subgroup):
    """Plot saved test estimates; error bars are sample SD over training seeds."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cv-project-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'yolo11n': '#0072B2', 'yolo11s': '#D55E00'}
    models = ['yolo11n', 'yolo11s']
    estimates = []
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300}):
        fig, ax = plt.subplots(figsize=(7.4, 4.4), layout='constrained')
        y = np.arange(len(names))
        for index, model in enumerate(models):
            means, deviations = [], []
            for name in names:
                rr = [r for r in classes if r['model'] == model and r['class_name'] == name]
                values = [float(r['ap50']) for r in rr]
                assert len(values) == 3
                means.append(statistics.mean(values)); deviations.append(statistics.stdev(values))
                estimates.append(dict(figure='class_ap50', model=model, group=name,
                                      support=int(rr[0]['support']), mean=means[-1], sample_sd=deviations[-1]))
            ax.barh(y + (index - .5) * .36, means, height=.34, xerr=deviations,
                    color=colors[model], label=model.replace('yolo', 'YOLO'),
                    error_kw=dict(ecolor='#333333', elinewidth=.8, capsize=2))
        support = {r['class_name']:r['support'] for r in classes}
        ax.set_yticks(y, [f'{name} (n={support[name]})' for name in names])
        ax.invert_yaxis()
        ax.set(xlim=(0, 1), xlabel='Test AP50', ylabel='Class and test ground-truth count')
        ax.set_axisbelow(True); ax.grid(axis='x', alpha=.2)
        ax.legend(loc='lower right', frameon=False)
        fig.savefig(FIG/'class_ap50_comparison.png'); plt.close(fig)

        sizes = ['small', 'medium', 'large']
        fig, ax = plt.subplots(figsize=(7.4, 2.8), layout='constrained')
        x = np.arange(len(sizes))
        for index, model in enumerate(models):
            means, deviations = [], []
            for size in sizes:
                rr = [r for r in subgroup if r['model'] == model and r['dimension'] == 'size'
                      and r['group'] == size and r['class_name'] == 'ALL']
                values = [float(r['recall']) for r in rr]
                assert len(values) == 3
                means.append(statistics.mean(values)); deviations.append(statistics.stdev(values))
                estimates.append(dict(figure='size_recall', model=model, group=size,
                                      support=int(rr[0]['support']), mean=means[-1], sample_sd=deviations[-1]))
            bars = ax.bar(x + (index - .5) * .32, means, width=.30, yerr=deviations,
                          color=colors[model], label=model.replace('yolo', 'YOLO'),
                          error_kw=dict(ecolor='#333333', elinewidth=.8, capsize=3))
            ax.bar_label(bars, labels=[f'{v:.3f}' for v in means], padding=7, fontsize=9)
        support = {r['group']:r['support'] for r in subgroup
                   if r['dimension'] == 'size' and r['class_name'] == 'ALL'}
        ax.set_xticks(x, [f'{size.title()} (n={support[size]})' for size in sizes])
        ax.set(ylim=(0, 1), ylabel='Test recall at IoU >= 0.50')
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper left', frameon=False, ncol=2)
        fig.savefig(FIG/'size_recall_comparison.png'); plt.close(fig)
    write_csv(OUT/'plot_estimates.csv', estimates)


def failure_figures(summary):
    caches = {m:json.loads((OUT/f'cache/{m}_seed42_test.json').read_text()) for m in ['yolo11n','yolo11s']}
    objects = read('test_objects.csv')
    # Fixed, visually inspected examples: distinct small miss, occluded miss and localization FP.
    cases = [('small_miss','test__image182.jpg',6,(330,30,610,290)),
             ('occluded_miss','test__image1037.jpeg',9,(260,145,610,410)),
             ('localization','test__image1125.jpg',9,(260,80,960,700))]
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21) if sys.platform=='darwin' else ImageFont.load_default(size=21)
    evidence = []
    for case, filename, g, crop in cases:
        canvas=Image.new('RGB',(1120,390),'white')
        for col,m in enumerate(['yolo11n','yolo11s']):
            im=next(x for x in caches[m]['images'] if Path(x['image']).name==filename)
            gt=im['targets'][g];cls=int(gt[4]);threshold=summary['protocol']['thresholds'][m]
            pred=np.asarray(im['predictions'],float).reshape(-1,6)
            kept,pairs=match_detections(pred,im['targets'],threshold)
            matched=next((p for p,idx in pairs if idx==g),None)
            same=[int(p) for p in kept if int(pred[p,5])==cls]
            ious=box_iou(pred[same,:4],np.array([gt[:4]]))[:,0] if same else np.array([])
            best=same[int(ious.argmax())] if len(ious) else None
            best_iou=float(ious.max()) if len(ious) else 0
            source=Image.open(ROOT/im['image']).convert('RGB')
            x0,y0,x1,y1=crop;x1=min(x1,source.width);y1=min(y1,source.height)
            panel=source.crop((x0,y0,x1,y1));panel.thumbnail((544,335))
            sx=panel.width/(x1-x0);sy=panel.height/(y1-y0)
            draw=ImageDraw.Draw(panel)
            def box(b,color):
                draw.rectangle(((b[0]-x0)*sx,(b[1]-y0)*sy,(b[2]-x0)*sx,(b[3]-y0)*sy),outline=color,width=4)
            # Display best same-class prediction only if it overlaps this target.
            if best is not None and best_iou>0:box(pred[best], '#00e5ff')
            box(gt,'#ffe600')
            x=col*560+(550-panel.width)//2;y=45+(335-panel.height)//2
            canvas.paste(panel,(x,y));d=ImageDraw.Draw(canvas)
            status='TP' if matched is not None else 'MISS'
            d.text((col*560+12,8),f'{m}, seed 42 | {status} | best IoU {best_iou:.2f}',font=font,fill='black')
            o=next(o for o in objects if o['image']==im['image'] and int(o['target_index'])==g)
            evidence.append(dict(case=case,model=m,seed=42,image=im['image'],target_index=g,
                                 source_image=o['source_image'],source_line=int(o['source_line']),
                                 class_name=o['class_name'],target=gt,area_640=float(o['area_640']),
                                 visibility=o['visibility'],confidence=threshold,matched=matched is not None,
                                 best_same_class_prediction=pred[best].tolist() if best is not None else None,
                                 best_iou=best_iou,
                                 best_prediction_matched_elsewhere=any(p==best for p,idx in pairs) if best is not None else False))
        canvas.save(FIG / f'{case}.png')
    (OUT/'failure_examples.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def main():
    TEX.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    summary=json.loads((OUT/'test_summary.json').read_text())
    comp=summary['comparison'];names=summary['class_names']
    rows=[]
    for label,key in [('mAP@0.50:0.95','map50_95'),('mAP@0.50','map50'),('Macro precision','macro_precision'),('Macro recall','macro_recall'),('Macro-F1','macro_f1'),('Macro-F1 (reference confidence 0.25)','macro_f1_ref025')]:
        rows.append([label]+[f"{r[key+'_mean']:.4f} $\\pm$ {r[key+'_std']:.4f}" for r in comp])
    table('aggregate.tex',['Test metric','YOLO11n','YOLO11s'],rows)
    rows=[]
    for r in summary['runs']:
        rows.append([r['model'],r['seed']]+[f'{r[k]:.4f}' for k in ['map50_95','map50','macro_precision','macro_recall','macro_f1']])
    table('seeds.tex',['Model','Seed','mAP50:95','mAP50','Macro P','Macro R','Macro-F1'],rows)
    rows=[]
    for m in ['yolo11n','yolo11s']:
        l=summary['latency'][m];c=next(r for r in summary['complexity'] if r['model']==m)
        train=json.loads((ROOT/f'reports/training/{m}_summary.json').read_text())
        rows.append([m,f"{c['parameters']/1e6:.2f}",f"{c['checkpoint_bytes']/1e6:.2f}",f"{l['median_ms']:.2f}",f"{l['p25_ms']:.2f}--{l['p75_ms']:.2f}",pm([r['train_seconds']/60 for r in train['runs']],1)])
    table('runtime.tex',['Model','Params (M)','Weights (MB)','CPU ms','IQR (ms)','Train min/seed'],rows)
    classes=read('per_class.csv');train_counts={r['class_name']:int(r['boxes']) for r in csv.DictReader((ROOT/'reports/dataset/class_counts.csv').open()) if r['split']=='train'}
    rows=[]
    for name in names:
        rr=[r for r in classes if r['class_name']==name];n=[r for r in rr if r['model']=='yolo11n'];s=[r for r in rr if r['model']=='yolo11s']
        rows.append([esc(name),train_counts[name],n[0]['support']]+[pm([float(r[k]) for r in group]) for k,group in [('ap50',n),('ap50',s),('recall',n),('recall',s)]])
    table('classes.tex',['Class','Train','Test','AP50 (n)','AP50 (s)','Recall (n)','Recall (s)'],rows)
    subgroup=read('subgroup_recall.csv');rows=[]
    for dim,groups in [('size',['small','medium','large']),('visibility',['clear','occluded','uncertain']),('truncated',['yes','no','uncertain'])]:
        for group in groups:
            rr=[r for r in subgroup if r['dimension']==dim and r['group']==group and r['class_name']=='ALL']
            rows.append([f'{dim}: {group}',rr[0]['support']]+[pm([float(r['recall']) for r in rr if r['model']==m]) for m in ['yolo11n','yolo11s']])
    table('subgroups.tex',['Object subset','Count','Recall (n)','Recall (s)'],rows)
    dist=read('visibility_size_distribution.csv');rows=[]
    for v in ['clear','occluded','uncertain']:rows.append([v]+[next(r['support'] for r in dist if r['visibility']==v and r['size']==s) for s in ['small','medium','large']])
    table('visibility_sizes.tex',['PPE visibility','Small','Medium','Large'],rows)
    rows=[]
    for name in names:
        for size in ['small','medium','large']:
            rr=[r for r in subgroup if r['dimension']=='size' and r['group']==size and r['class_name']==name]
            if rr:rows.append([esc(name),size,rr[0]['support']]+[pm([float(r['recall']) for r in rr if r['model']==m]) for m in ['yolo11n','yolo11s']])
    table('size_classes.tex',['Class','Size','Count','Recall (n)','Recall (s)'],rows)
    comparison_figures(names, classes, subgroup)
    evidence=failure_figures(summary)
    n,s=comp;ratio=summary['latency']['yolo11s']['median_ms']/summary['latency']['yolo11n']['median_ms']
    (TEX/'measured_values.tex').write_text(f'\\newcommand{{\\APgain}}{{{s["map50_95_mean"]-n["map50_95_mean"]:.4f}}}\n'+f'\\newcommand{{\\Fgain}}{{{s["macro_f1_mean"]-n["macro_f1_mean"]:.4f}}}\n'+f'\\newcommand{{\\LatencyRatio}}{{{ratio:.2f}}}\n')
    # Bundle editable sources independently of the repository's directory layout.
    with zipfile.ZipFile(ROOT/'planning/results_checkpoint_latex.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(TEX.glob('*.tex')):
            content = path.read_text().replace('../../reports/results_checkpoint/figures/', 'figures/')
            archive.writestr(path.name, content)
        for path in sorted(FIG.glob('*.png')):
            archive.write(path, 'figures/'+path.name)
        archive.writestr('README.txt', 'Compile results_checkpoint.tex with tectonic, or run pdflatex twice.\n'
                         'All tables and figures are included. Results and analysis are in English.\n')
    print(f'Generated tables and {len(evidence)//2} failure figures in {TEX.relative_to(ROOT)}')


if __name__=='__main__':main()
