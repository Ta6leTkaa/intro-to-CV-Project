# Construction-PPE audit and preparation

The full archive was checked: 1416 images and 11 classes. All images are readable and
have matching labels. Coordinates are valid, and there are no empty annotations.
After removing one exact duplicate Person box in train/image187.jpg, **11520 boxes** remain.
Ten label files without matching images were excluded from the prepared copy.
See [annotation_issues.csv](annotation_issues.csv).

## Dataset splits

| Split | Original | Prepared |
|---|---:|---:|
| train | 1132 | 1265 |
| val | 143 | 73 |
| test | 141 | 78 |

The original splits contain related scenes and transformed copies, including rotations,
brightness changes, and similar poses of the same person against the same background.
There are no exact duplicates of decoded images. A 63-bit pHash with Hamming distance ≤6
identified 91 pairs across splits; visual review confirmed all of them as related.
Review of the full image set also produced 14 scene groups. Images are grouped using
these decisions and exact matches. Each group is assigned entirely to the original split
containing most of its images; ties favour train, then val, then test. This rule does not
use model metrics. A total of 149 images were moved, and none were removed.

The [manifest](split_manifest.csv) records original and final splits, groups, dimensions,
and hashes. The [pair review](leakage_review.csv) and [scene groups](scene_groups.csv)
preserve the review decisions. No known related pairs cross the final splits. However,
without official scene IDs, visual review and pHash cannot guarantee that every relationship
has been found. Some test images remain related to each other, reducing the effective number
of independent examples. The small validation and test sets, along with rare classes,
require caution when interpreting AP. The split sizes differ from the PDF because of the
leakage corrections anticipated in that plan.

## Classes and imbalance

Number of **boxes**, rather than images:

| ID | Class | Train | Val | Test |
|---|---|---:|---:|---:|
| 0 | helmet | 1454 | 119 | 161 |
| 1 | gloves | 1351 | 37 | 57 |
| 2 | vest | 1376 | 92 | 150 |
| 3 | boots | 1413 | 40 | 144 |
| 4 | goggles | 501 | 8 | 9 |
| 5 | none | 677 | 69 | 51 |
| 6 | Person | 1884 | 162 | 198 |
| 7 | no_helmet | 400 | 45 | 40 |
| 8 | no_goggle | 337 | 41 | 33 |
| 9 | no_gloves | 442 | 56 | 58 |
| 10 | no_boots | 88 | 4 | 23 |

All 11 classes appear in every split. In training, the most frequent class has roughly
21.4 times as many boxes as the least frequent class. Validation and test sets are not
rebalanced, and rare images are not duplicated. The team should report per-class AP and
macro F1, as specified in the plan.

- [class_counts.csv](class_counts.csv): box and image counts by class and split.
- [class_distribution.png](class_distribution.png): training box distribution.
- [size_counts.csv](size_counts.csv) and [size_distribution.png](size_distribution.png): object sizes.

Size is determined by box area after letterboxing to 640 pixels, using the scale
min(640/W, 640/H): small <32², medium ≥32² and <96², large ≥96². Padding does not change
box area. The original images were not resized; these sizes were calculated for analysis.

## Annotation limitations

Twenty boxes per class were reviewed before training. Findings are recorded in
[class_review.csv](class_review.csv), with examples in [visual_samples.csv](visual_samples.csv).
This is a limited, deterministic sample, rather than a complete manual audit of every box.
In the reviewed examples, `none` labels a torso in ordinary clothing; it does not mean
background. The original name is retained. The `no_*` classes label the head, eye region,
hands, or lower-body region without the corresponding PPE. There is no separate `no_vest`.
Some goggles annotations include ordinary glasses or sunglasses, and some boots annotations
include ordinary footwear. The dataset also includes sports and everyday photographs,
as well as overlapping Person boxes. It does not establish whether a worker's equipment
meets safety certification requirements. Ambiguous annotations are retained and documented;
classes have not been renamed.

## Occlusion review and handoff

[occlusion_review.csv](occlusion_review.csv) contains 521 test objects from helmet, gloves,
vest, boots, and goggles. All objects were visually reviewed by the assistant at Andrei's request,
using original-image contact sheets and individual contextual crops. No model predictions were used.
`member1_tag` and `final_tag` contain this review; `truncated` and explanatory `notes` are complete.
Results: **259 clear, 72 occluded, 190 uncertain**. Truncation: **24 yes, 309 no, 188 uncertain**.
Counts by class and review provenance are in [occlusion_summary.json](occlusion_summary.json).

For the clear/occluded comparison, exclude `final_tag=uncertain` and report group sizes.
Ambiguous source boxes, blur, and extreme transformations account for many uncertain cases.
These labels assess visibility; source classes and coordinates were not corrected.
This is a single assistant review. Member 4's independent assessment is still empty, so these
results must not be described as agreement between two human reviewers. The original team plan
requires that joint review; its status and protocol are recorded in
[member_1.md](../../planning/member_1.md).

CSV paths refer to the source dataset. Prepared image names follow
`<original_split>__<filename>`, with the folder taken from the manifest's `split` field.
The occlusion CSV's `line` field is the source annotation line number, starting at 1.
The generated YAML is `data/prepared/construction-ppe/dataset.yaml`.
Reproduction commands are in the [root README](../../README.md).

## Reproducibility checks

All 1416 prepared image/label pairs, box counts, class IDs, and unchanged image bytes were
verified. Scene groups and decoded-image hashes do not cross splits. A second run produced
byte-identical manifests, statistics tables, summary, and the then-preliminary occlusion CSV.
These checks validate dataset preparation; no model was trained.

The completed assistant occlusion review was also checked against all source annotation lines;
a subsequent preparation run preserved the completed CSV byte for byte.
