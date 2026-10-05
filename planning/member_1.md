# Member 1 — Andrei Dubinin

Your responsibilities in the team plan are to audit Construction-PPE, prepare the splits,
and produce dataset statistics. You also review occlusion labels with Member 4.
This checklist covers those responsibilities.

## Dataset preparation — completed

- [x] Download the full dataset and record its source, license, and checksum.
- [x] Retain all 11 original classes and IDs; verify them against the official YAML.
- [x] Check image readability, image/label pairs, YOLO format, and bounding box coordinates.
- [x] Review 20 boxes per class and document annotation ambiguities.
- [x] Check similar images across splits and visually review the scenes.
- [x] Group the related images found during review and save the final split manifest.
- [x] Count boxes and images by class, analyse object sizes and imbalance, and create two plots.
- [x] Create the train/val/test folders and `dataset.yaml`; verify reproducibility.

Result: **1416 images, 11520 boxes, 11 classes**. Split sizes: **1265 / 73 / 78**.
Ten label files without images were excluded; one exact duplicate box was removed from
the prepared copy. The source data is preserved. Details and limitations:
[dataset audit report](../reports/dataset/README.md).

## Occlusion labels — assistant review completed

At Andrei's request, the assistant reviewed all **521 test PPE objects** using the original
image contact sheets and individual contextual crops, without model predictions.
`member1_tag` and `final_tag` contain the assistant's judgments; `truncated` and explanatory
`notes` are filled in for every object. Results: **259 clear, 72 occluded, 190 uncertain**.
There are 24 objects marked truncated, 188 with uncertain truncation, and 309 marked not truncated.
See [occlusion_review.csv](../reports/dataset/occlusion_review.csv) and
[occlusion_summary.json](../reports/dataset/occlusion_summary.json).

Use `clear` when the object is visible without noticeable obstruction, `occluded` when
another object or person hides part of it, and `uncertain` when the image does not support
a confident judgment. Low resolution, blur, extreme image transformations, and source boxes
that do not clearly contain the named object are reasons for uncertainty. Truncation is
recorded separately and includes clipping at the original photo boundary inside black padding
or at a transformed image edge. An arm folded across a vest counts as occlusion of that vest.
Source class labels and box coordinates are unchanged.

For evaluation, compare only `clear` and `occluded`; exclude `uncertain` and report the numbers
of objects in each group. Report truncation separately. The labels describe visibility,
not whether PPE meets safety requirements.

**Review provenance:** this is one assistant review, not agreement between two human reviewers.
The team PDF specifies a joint review by Members 1 and 4. `member4_tag` is deliberately empty;
that independent review has not occurred. If the team follows the original two-reviewer protocol,
Member 4 should label the same objects independently before viewing predictions, then the team
should resolve disagreements and update `final_tag`. The assistant results are ready for handoff
with this limitation recorded.

## Team handoff

- Member 2: `data/prepared/construction-ppe/dataset.yaml` and the commands in the root README.
  Use this split for every experiment; do not create a new random split.
- Member 3: `split_manifest.csv`, `class_counts.csv`, and `size_counts.csv`;
  provide `occlusion_review.csv` and `occlusion_summary.json`, including their review provenance.
- Member 4: the plots and limitations described in the dataset audit report.

The training data loader handles RGB conversion, letterboxing to 640 pixels without
stretching, and augmentation of training images only. There is no need to save every image
again at 640×640. Preparation adds no augmentations. The source archive already contains
transformed copies, which were considered during grouping. Training setup, model comparison,
threshold selection, metrics, failure analysis, and the presentation remain assigned to
the team members listed in the PDF.
