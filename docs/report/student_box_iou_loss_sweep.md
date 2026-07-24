# Box IoU loss sweep

## Question

The native student predicts one normalized evidence box from the language state immediately before
the gold answer. The baseline objective adds SmoothL1 coordinate regression to generalized IoU
(GIoU). GIoU rewards overlap and penalizes empty space in the enclosing box, but it does not
directly optimize center distance or aspect-ratio agreement. Those omissions may matter for thin
text lines, table cells, chart marks, and spatially separated document evidence.

The O03 entry in the
[`frontier method survey`](frontier_method_survey.md) therefore remains an ablation rather than a
default. The experiment compares GIoU with distance IoU (DIoU) and complete IoU (CIoU) while
retaining the same SmoothL1 term.

## Objectives

For predicted box \(B\), target \(B^\*\), and smallest enclosing box \(C\), define
\(\rho^2\) as squared center distance and \(c^2\) as the squared diagonal of \(C\).

\[
L_{\mathrm{DIoU}} =
1-\operatorname{IoU}(B,B^\*) + \frac{\rho^2(B,B^\*)}{c^2}.
\]

CIoU adds an aspect-ratio term:

\[
L_{\mathrm{CIoU}} =
L_{\mathrm{DIoU}} + \alpha v,
\qquad
\alpha = \frac{v}{1-\operatorname{IoU}(B,B^\*)+v},
\qquad
v = \frac{4}{\pi^2}
\left(
\arctan\frac{w^\*}{h^\*} -
\arctan\frac{w}{h}
\right)^2.
\]

The executable training loss is always
\(L_{\mathrm{SmoothL1}} + L_{\mathrm{IoU\ family}}\). Configure its IoU-family term with:

```yaml
training:
  pretraining:
    box_iou_loss: giou
```

Supported values are `giou`, `diou`, and `ciou`. Exact-match boxes have zero IoU-family loss. CIoU
adds a positive penalty over DIoU when boxes share a center but disagree in aspect ratio.
Training records include `train/box_iou_loss` alongside the box-regression scalar and cumulative
student FLOPs.

## Paired experiment

[`configs/sub1b_box_iou_loss_sweep.yaml`](../../configs/sub1b_box_iou_loss_sweep.yaml) compiles
three objectives by three paired stochastic replicates. Every arm keeps constant:

- model architecture, initialization, tokenizer, data mixture, and selected teacher targets;
- authored train and heldout documents plus public-data selection;
- augmentation, sampler, optimizer, SFT, RLVR, and evaluation seeds;
- every loss weight and curriculum boundary;
- post-training targets, rewards, rollout policy, and deployment gates.

The nominal 20B effective-token pretraining dose is converted with the compute suite's default
896-pixel, 64-latent reference profile and its 1,024-text-token sample shape. The resulting budget
is `165669831748966989312` algorithmic student FLOPs. Each arm uses a student-FLOP learning-rate
schedule and stops after crossing that same budget, with at most one optimizer-update overshoot.
The curriculum advances by training-compute fraction rather than token fraction, so its loss-weight
boundaries remain matched under the FLOP stop.

Inspect or execute the nine runs:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_box_iou_loss_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_box_iou_loss_sweep.yaml
```

W&B uses group `docvlm-box-iou-loss-ablation` and tags every run with
`box-iou-loss-ablation`.

## Resume contract

The selected IoU-family objective is stored in the pretraining supervision contract. Resume rejects
a changed objective before loading optimizer state. This prevents one checkpoint trajectory from
mixing GIoU, DIoU, and CIoU updates.

## Promotion rule

DIoU or CIoU is promoted only when its paired heldout interval improves grounding or evidence
localization without failing OCR, table, chart, multilingual, reliability, or deployment gates.
Report evidence-count and degradation slices, train-minus-heldout gaps, and cumulative
`train/student_flops_seen`. Lower box training loss, larger gradients, or better train-set IoU alone
is insufficient. If the heldout interval crosses zero, GIoU remains the conservative default.

This configuration is an executable experiment contract, not evidence that DIoU or CIoU is already
better. A quality claim requires the completed GPU runs and aggregated paired intervals.
