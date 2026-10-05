"""Audit all labels, review leakage candidates, and prepare YOLO dataset splits."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cv-project-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cv-project-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/construction-ppe'
OUT = ROOT / 'reports/dataset'
NAMES = ['helmet', 'gloves', 'vest', 'boots', 'goggles', 'none', 'Person',
         'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']
SPLITS = ['train', 'val', 'test']
COLORS = ['#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#06b6d4',
          '#3b82f6','#8b5cf6','#d946ef','#ec4899','#f43f5e']


def csv_write(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not path.exists(): return []
    with path.open(newline='') as f: return list(csv.DictReader(f))


def image_hash(im):
    small = np.asarray(im.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    t = np.arange(32)
    transform = np.cos(np.pi * (2*t[None,:] + 1) * np.arange(8)[:,None] / 64)
    low = (transform @ small @ transform.T).flatten()[1:]
    return sum(int(x > np.median(low)) << i for i,x in enumerate(low))


def scan():
    rows, boxes, issues = [], {}, []
    if not (RAW / 'images/train').is_dir():
        raise SystemExit('Run scripts/download_dataset.py first.')
    for split in SPLITS:
        images = sorted(p for p in (RAW / 'images' / split).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.bmp'])
        stems = {p.stem for p in images}
        for label in sorted((RAW / 'labels' / split).glob('*.txt')):
            if label.stem not in stems: issues.append({'image': str(label.relative_to(RAW)), 'line': '', 'problem':'orphan_label'})
        for p in images:
            key = p.relative_to(RAW).as_posix()
            label = RAW / 'labels' / split / f'{p.stem}.txt'
            try:
                with Image.open(p) as src:
                    src.load()
                    im = src.convert('RGB')
                    w,h = im.size
                    orientation = src.getexif().get(274,1)
                    if orientation not in (None,1):
                        issues.append({'image':key,'line':'','problem':f'EXIF orientation {orientation}: review coordinates'})
                    pixels = hashlib.sha256(f'{w}x{h}:'.encode() + im.tobytes()).hexdigest()
                    phash = image_hash(im)
            except Exception as exc:
                issues.append({'image':key,'line':'','problem':f'unreadable image: {exc}'})
                continue
            labels=[]
            if not label.exists():
                issues.append({'image':key,'line':'','problem':'missing_label'})
            else:
                for line_no,line in enumerate(label.read_text().splitlines(),1):
                    if not line.strip():continue
                    try:
                        parts=line.split()
                        if len(parts)!=5:raise ValueError('expected 5 fields')
                        cls=int(parts[0]); cx,cy,bw,bh=map(float,parts[1:])
                        if not 0<=cls<11:raise ValueError('invalid class ID')
                        if not all(math.isfinite(x) for x in (cx,cy,bw,bh)):raise ValueError('non-finite coordinate')
                        if not (0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1):raise ValueError('invalid normalized coordinate')
                        if min(cx-bw/2,cy-bh/2)<-1e-5 or max(cx+bw/2,cy+bh/2)>1+1e-5:raise ValueError('box crosses image boundary')
                        area=bw*w*bh*h*min(640/w,640/h)**2
                        labels.append({'class_id':cls,'cx':cx,'cy':cy,'w':bw,'h':bh,'line':line_no,
                                       'area_640':area,'size':'small' if area<32**2 else 'medium' if area<96**2 else 'large'})
                    except ValueError as exc:
                        issues.append({'image':key,'line':line_no,'problem':str(exc)})
            unique=[];seen=set()
            for box in labels:
                identity=tuple(box[k] for k in ['class_id','cx','cy','w','h'])
                if identity in seen:
                    issues.append({'image':key,'line':box['line'],'problem':'duplicate box annotation'})
                else:
                    seen.add(identity);unique.append(box)
            boxes[key]=unique
            rows.append({'image':key,'label':label.relative_to(RAW).as_posix(),'original_split':split,
                         'width':w,'height':h,'boxes':len(boxes[key]),'pixel_sha256':pixels,
                         'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest() if label.exists() else '',
                         'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'phash':f'{phash:016x}'})
    return rows,boxes,issues


def annotate(image_key, labels, highlight=None, width=600):
    with Image.open(RAW / image_key) as src:
        im=src.convert('RGB')
    im.thumbnail((width,420))
    draw=ImageDraw.Draw(im)
    for b in labels:
        if highlight is not None and b['line']!=highlight:continue
        x0=(b['cx']-b['w']/2)*im.width;y0=(b['cy']-b['h']/2)*im.height
        x1=(b['cx']+b['w']/2)*im.width;y1=(b['cy']+b['h']/2)*im.height
        color=COLORS[b['class_id']]
        draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
        draw.text((max(0,x0),max(0,y0-12)),f"{b['class_id']}:{NAMES[b['class_id']]} L{b['line']}",fill=color,stroke_width=1,stroke_fill='white')
    return im


def grid(items, path, columns=4, tile=(300,250)):
    canvas=Image.new('RGB',(columns*tile[0],math.ceil(len(items)/columns)*tile[1]),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(items):
        x=(i%columns)*tile[0];y=(i//columns)*tile[1]
        im=im.copy();im.thumbnail((tile[0]-10,tile[1]-44))
        canvas.paste(im,(x+5,y+40))
        draw.text((x+5,y+3),title,fill='black')
    canvas.save(path)


def candidates(rows):
    found=[]
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if left['original_split']==right['original_split']:continue
            exact=left['pixel_sha256']==right['pixel_sha256']
            distance=(int(left['phash'],16)^int(right['phash'],16)).bit_count()
            if exact or distance<=6:
                found.append({'left':left['image'],'right':right['image'],'distance':distance,
                              'kind':'exact' if exact else 'similar','decision':'same_scene' if exact else '', 'notes':''})
    return found


def main(apply=False):
    OUT.mkdir(parents=True,exist_ok=True)
    # Confirm documented IDs agree with the archive before any preparation.
    import re
    for config in (RAW/'data.yaml', ROOT/'data/upstream-construction-ppe.yaml'):
        found={int(i):name.strip() for i,name in re.findall(r'^  (\d+): (.+)$',config.read_text(),re.M)}
        if found != dict(enumerate(NAMES)):raise SystemExit(f'Unexpected class mapping: {config}')
    rows,boxes,issues=scan()
    csv_write(OUT/'annotation_issues.csv',issues,['image','line','problem'])
    review_path=OUT/'leakage_review.csv'
    old={(r['left'],r['right']):r for r in read_csv(review_path)}
    pairs=candidates(rows)
    for pair in pairs:
        if pair['kind']!='exact' and (pair['left'],pair['right']) in old:
            prior=old[pair['left'],pair['right']]
            pair['decision']=prior['decision'];pair['notes']=prior['notes']
    csv_write(review_path,pairs,['left','right','distance','kind','decision','notes'])
    review_dir=OUT/'review_images';review_dir.mkdir(exist_ok=True)
    for offset in range(0,len(pairs),8):
        items=[]
        for j,pair in enumerate(pairs[offset:offset+8],offset):
            for side in ('left','right'):
                key=pair[side]
                with Image.open(RAW/key) as src: im=src.convert('RGB')
                items.append((im,f"Pair {j} ({pair['kind']}, d={pair['distance']})\n{key}"))
        grid(items,review_dir/f'leakage_{offset//8:03d}.jpg',columns=4,tile=(380,290))
    # Review examples: deterministic sampling from all classes, not predictions.
    visual=[]
    for cls in range(11):
        examples=[(key,b) for key in sorted(boxes) for b in boxes[key] if b['class_id']==cls]
        sample=examples[:20]
        items=[]
        for key,b in sample:
            with Image.open(RAW/key) as src:
                im=src.convert('RGB');w,h=im.size
                x0=max(0,int((b['cx']-b['w']/2)*w)-15);y0=max(0,int((b['cy']-b['h']/2)*h)-15)
                x1=min(w,math.ceil((b['cx']+b['w']/2)*w)+15);y1=min(h,math.ceil((b['cy']+b['h']/2)*h)+15)
                im=im.crop((x0,y0,x1,y1))
            items.append((im,f"{cls}:{NAMES[cls]} L{b['line']}\n{key}"))
            visual.append({'class_id':cls,'class_name':NAMES[cls],'image':key,'line':b['line']})
        if items:grid(items,review_dir/f'class_{cls:02d}.jpg')
    csv_write(OUT/'visual_samples.csv',visual,['class_id','class_name','image','line'])
    unresolved=[p for p in pairs if p['decision'] not in ('same_scene','different')]
    parent={r['image']:r['image'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):parent[find(b)]=find(a)
    # All exact copies and manually confirmed related images form indivisible groups.
    byhash=collections.defaultdict(list)
    for r in rows:byhash[r['pixel_sha256']].append(r['image'])
    for group in byhash.values():
        for key in group[1:]:union(group[0],key)
    for p in pairs:
        if p['decision']=='same_scene':union(p['left'],p['right'])
    scene_rows=read_csv(OUT/'scene_groups.csv')
    scene_members=collections.defaultdict(list)
    for item in scene_rows:
        if item['image'] not in parent:raise ValueError('Unknown image in scene review')
        scene_members[item['scene_group']].append(item['image'])
    for group in scene_members.values():
        for key in group[1:]:union(group[0],key)
    groups=collections.defaultdict(list)
    for r in rows:groups[find(r['image'])].append(r)
    for group in groups.values():
        # Majority split minimizes moves; deterministic ties prefer train, then val.
        counts=collections.Counter(r['original_split'] for r in group)
        target=max(SPLITS,key=lambda split:(counts[split],-SPLITS.index(split)))
        for r in group:r['split']=target;r['group']=min(x['image'] for x in group)
    fields=['image','label','original_split','split','group','width','height','boxes','pixel_sha256','file_sha256','label_sha256','phash']
    csv_write(OUT/'split_manifest.csv',rows,fields)
    stats=[];size_rows=[]
    for split in ['all']+SPLITS:
        subset=[r for r in rows if split=='all' or r['split']==split]
        for cls,name in enumerate(NAMES):
            selected=[b for r in subset for b in boxes[r['image']] if b['class_id']==cls]
            stats.append({'split':split,'class_id':cls,'class_name':name,'boxes':len(selected),
                          'images':sum(any(b['class_id']==cls for b in boxes[r['image']]) for r in subset)})
            for size in ('small','medium','large'):
                size_rows.append({'split':split,'class_id':cls,'class_name':name,'size':size,
                                  'boxes':sum(b['size']==size for b in selected)})
    csv_write(OUT/'class_counts.csv',stats,['split','class_id','class_name','boxes','images'])
    csv_write(OUT/'size_counts.csv',size_rows,['split','class_id','class_name','size','boxes'])
    counts=[next(x['boxes'] for x in stats if x['split']=='train' and x['class_id']==cls) for cls in range(11)]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(NAMES,counts,color=COLORS)
    ax.set_ylabel('Training boxes');ax.tick_params(axis='x',labelrotation=45)
    fig.tight_layout();fig.savefig(OUT/'class_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));bottom=np.zeros(11)
    for size,color in zip(('small','medium','large'),('#3b82f6','#f59e0b','#22c55e')):
        vals=np.array([next(x['boxes'] for x in size_rows if x['split']=='all' and x['class_id']==cls and x['size']==size) for cls in range(11)])
        ax.bar(NAMES,vals,bottom=bottom,label=size,color=color);bottom+=vals
    ax.set_ylabel('Boxes at 640-pixel input');ax.tick_params(axis='x',labelrotation=45);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'size_distribution.png',dpi=160);plt.close(fig)
    prepared=ROOT/'data/prepared/construction-ppe'
    blocking=[i for i in issues if i['problem'] not in ('orphan_label','duplicate box annotation')]
    semantic_review=read_csv(OUT/'class_review.csv')
    semantic_ready={int(r['class_id']) for r in semantic_review if r['status']=='sample_reviewed'}==set(range(11))
    ready=not blocking and not unresolved and semantic_ready
    summary={'source_images':dict(collections.Counter(r['original_split'] for r in rows)),
        'prepared_images':dict(collections.Counter(r['split'] for r in rows)),
        'total_images':len(rows),'total_boxes':sum(len(x) for x in boxes.values()),
        'empty_labels':sum(not boxes[r['image']] for r in rows),'source_annotation_issues':len(issues),'blocking_annotation_issues':len(blocking),
        'ignored_orphan_labels':sum(i['problem']=='orphan_label' for i in issues),
        'removed_duplicate_boxes':sum(i['problem']=='duplicate box annotation' for i in issues),
        'classes_visually_reviewed':len(semantic_review),
        'exact_duplicate_groups':sum(len(v)>1 for v in byhash.values()),
        'cross_split_candidates':len(pairs),'unreviewed_candidates':len(unresolved),
        'moved_images':sum(r['split']!=r['original_split'] for r in rows),
        'manual_scene_groups':len(scene_members),'group_count':len(groups),
        'remaining_known_cross_split_pairs':sum(next(r['split'] for r in rows if r['image']==p['left'])!=next(r['split'] for r in rows if r['image']==p['right']) for p in pairs if p['decision']=='same_scene'),
        'train_max_min_class_ratio':max(counts)/min(x for x in counts if x>0) if any(counts) else None,
        'missing_train_classes':[NAMES[i] for i,c in enumerate(counts) if not c],
        'ready_for_training':ready and apply}
    if apply:
        if not ready:
            summary['ready_for_training']=False
            if prepared.exists():raise SystemExit('Audit no longer passes; remove stale prepared data before retrying.')
        else:
            if prepared.exists():shutil.rmtree(prepared)
            for split in SPLITS:
                (prepared/'images'/split).mkdir(parents=True)
                (prepared/'labels'/split).mkdir(parents=True)
            for r in rows:
                source=RAW/r['image'];name=f"{r['original_split']}__{source.name}"
                dest=prepared/'images'/r['split']/name
                shutil.copy2(source,dest)
                label_dest=prepared/'labels'/r['split']/(Path(name).stem+'.txt')
                source_lines=(RAW/r['label']).read_text().splitlines()
                label_dest.write_text('\n'.join(source_lines[b['line']-1] for b in boxes[r['image']])+'\n')
            yaml=[f'path: {prepared.as_posix()}','train: images/train','val: images/val','test: images/test','names:']
            yaml += [f'  {i}: {name}' for i,name in enumerate(NAMES)]
            (prepared/'dataset.yaml').write_text('\n'.join(yaml)+'\n')
    # Blank tags deliberately require human judgments, including Member 4 review.
    occpath=OUT/'occlusion_review.csv'
    old_occ={(r['image'],r['line']):r for r in read_csv(occpath)}
    occ=[];occdir=OUT/'occlusion_images';occdir.mkdir(exist_ok=True)
    for r in rows:
        if r['split']!='test':continue
        for b in boxes[r['image']]:
            if b['class_id']>=5:continue
            row={'image':r['image'],'line':b['line'],'class_name':NAMES[b['class_id']],
                 'member1_tag':'','member4_tag':'','final_tag':'','truncated':'','notes':''}
            prior=old_occ.get((r['image'],str(b['line'])))
            if prior:
                for field in ['member1_tag','member4_tag','final_tag','truncated','notes']:row[field]=prior[field]
            occ.append(row)
    csv_write(occpath,occ,['image','line','class_name','member1_tag','member4_tag','final_tag','truncated','notes'])
    keys=sorted({r['image'] for r in occ})
    for old_image in occdir.glob('test_*.jpg'):old_image.unlink()
    for offset in range(0,len(keys),12):
        items=[(annotate(key,[b for b in boxes[key] if b['class_id']<5]),key) for key in keys[offset:offset+12]]
        grid(items,occdir/f'test_{offset//12:03d}.jpg',columns=3,tile=(520,440))
    summary['occlusion_objects']=len(occ)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if apply and not ready:raise SystemExit('Resolve blocking annotation issues, class reviews, and leakage reviews before preparing the training split.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Create training folders after the audit and manual leakage review pass')
    main(parser.parse_args().apply)
