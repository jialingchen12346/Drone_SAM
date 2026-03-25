# Model Redesign Note

Date: 2026-03-23

## Why Redesign Now

The previous headline claim `68.62 > 66.10` is not strict enough to support a
paper-ready comparison, because the two numbers are not produced under the same
metric definition.

Observed issue:

- The local CACAF evaluation uses `nanmean` over per-class IoU, so absent
  classes are skipped.
- The local MM SAM-Adapter baseline JSON reports `66.10`, and its class list
  includes `Bicycle = 0.0`, which implies a 14-class average with the empty
  class counted as zero.
- Under the CACAF per-class IoU listed in the session snapshot, `68.62` is the
  13-class average with absent classes skipped. If the empty class is scored as
  zero, the same per-class list drops to about `63.72`.

Consequence:

- The current result can still be used as an internal reference.
- It should not be used as the main evidence for outperforming MM SAM-Adapter.
- A new training round should be evaluated under one explicit protocol from the
  beginning.

## Current Model Weaknesses

The existing design already proved that multimodal fusion helps, but it is not
strong enough as the next-round innovation core.

### 1. CACAF is too coarse in how it models modality reliability

- MQA predicts one scalar per modality per scale.
- This is global weighting, not region-aware weighting.
- It cannot express the common FMB case where thermal is more reliable only on
  small objects or low-illumination regions.

### 2. f1 does not really participate in cross-modal interaction

- The highest-resolution stage skips cross-attention entirely for memory
  reasons.
- This means the level that matters most for edges and small objects receives
  only weighted averaging, not actual cross-modal correction.

### 3. HGSOAD is still a gated U-Net, not a genuinely new decoder

- SAGU is useful but structurally close to channel attention.
- The decoder does not explicitly separate detail recovery from semantic
  context aggregation.
- "small-object-aware" is partially true empirically, but the architecture
  story is not sharp enough for a full restart.

## Recommended Redesign Direction

Keep the asymmetric dual encoder, but replace the innovation center from
"global modal weighting + gated decoder" to:

1. Region-aware reliability fusion
2. Detail-semantic decoupled decoding
3. Unified low-cost multiscale interaction

### Proposed vNext architecture

Working name: `R2F2-D2 Decoder`

- `R2F2`: Region-Reliability Fusion
- `D2`: Detail-to-Depth decoder

Pipeline:

1. RGB encoder: keep SAM2 Hiera + adapter
2. Aux encoder: keep ConvNeXt-Tiny for thermal
3. Replace CACAF with a lighter region-aware fusion module
4. Replace HGSOAD with a dual-path decoder

## Innovation Point A: Region-Reliability Fusion

Target: replace scalar modal weighting with spatially varying reliability.

Minimal version:

- Align RGB/Aux features to a shared width at each scale
- Predict a 2-channel reliability map `R in [B, 2, H, W]`
- Normalize with softmax over modalities
- Fuse with `R_rgb * F_rgb + R_aux * F_aux`

Then add low-cost interaction:

- For `f1/f2`: use depthwise 3x3 cross-gating or local-window mixing instead of
  full cross-attention
- For `f3/f4`: keep compressed bidirectional interaction, but operate after
  channel reduction

Why this is stronger:

- It directly matches the problem statement: reliability varies by region, not
  only by image.
- It removes the weakest part of the current story, namely one scalar deciding
  an entire scale.
- It lets shallow features participate without quadratic attention cost.

## Innovation Point B: Detail-Semantic Decoupled Decoder

Target: replace the current single U-Net stream with two explicit roles.

Structure:

- Detail branch:
  - starts from `f1/f2`
  - preserves edges, poles, traffic lights, thin structures
- Semantic branch:
  - starts from `f3/f4`
  - aggregates object- and scene-level context
- Guided merge:
  - semantic branch predicts guidance maps
  - detail branch uses them to suppress noise and keep discriminative boundaries

Minimal implementation:

- top-down semantic path on `f4 -> f3 -> f2`
- shallow detail path on `f1` with one refinement block
- merge at stride 4, then predict logits

Why this is stronger:

- It gives a clearer explanation for small-object handling.
- It avoids claiming that generic channel gating alone solves small objects.
- It remains trainable on current hardware.

## Training Policy for the Redesign Round

Do not optimize around the previous `68.62` number.

New protocol:

1. Choose one official comparison metric and state it explicitly:
   - `skip-absent` or
   - `absent-as-zero`
2. Re-evaluate both the baseline and new models under the same script
3. Use the same test-time resize and label mapping for all methods
4. Only compare per-class numbers after this normalization

Recommended choice:

- For strict comparability with the existing MM SAM-Adapter JSON, treat absent
  classes as `0.0` during averaged metrics unless the baseline code proves
  otherwise.

## Immediate Next Steps

1. Re-run the current best checkpoint with the updated evaluator and record both
   metric variants.
2. Re-evaluate the MM SAM-Adapter baseline under the same reporting template.
3. Implement the first redesign prototype:
   - region-reliability fusion
   - dual-path decoder
   - keep encoder stack unchanged
4. Train one controlled prototype before adding any extra losses or tricks.
