# Decisions Log

Use this file to capture decisions that should survive across sessions.

## Template

### 2026-03-23
- Decision: Stop using `68.62 > 66.10` as the main strict comparison claim until FMB metric aggregation is unified.
- Context: The current CACAF evaluation skips absent classes with `nanmean`, while the saved MM SAM-Adapter baseline JSON likely averages over all 14 classes with the empty `Bicycle` class scored as `0.0`.
- Evidence: The baseline JSON gives `mIoU=66.10` with `IoU.Bicycle=0.0`; the recorded CACAF `68.62` corresponds to a 13-class average with absent classes skipped, and the same per-class list would drop to about `63.72` if the absent class were counted as zero.
- Consequence: The next-round work should prioritize unified evaluation and a redesigned architecture over incremental tuning of the current CACAF-HGSOAD result.

### 2026-03-23
- Decision: Keep the encoder stack fixed for the first redesign round and move the innovation center to fusion and decoding.
- Context: The current SAM2 Hiera + ConvNeXt-Tiny stack is already functional and not the main source of the paper-risk.
- Evidence: The weakest points are in the current innovation story: global scalar modal weighting, no true high-resolution interaction at `f1`, and a decoder that is still close to a gated U-Net.
- Consequence: The first prototype should replace CACAF/HGSOAD with region-reliability fusion and a detail-semantic decoupled decoder before exploring broader backbone changes.
