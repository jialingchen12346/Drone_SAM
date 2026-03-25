# Session Handoff

Copy the latest high-signal state here before ending a long session.

## Current Objective

- Redesign the fusion and decoder modules, then retrain under a unified FMB metric definition.

## Last Completed Work

- Verified that the current `68.62` claim is not strictly comparable to the local MM SAM-Adapter `66.10` because the mIoU averaging policy for absent classes is different.
- Updated `segmentation/eval_fullres.py` and `segmentation/train_cacaf.py` so evaluation can explicitly report both `skip-absent` and `absent-as-zero` variants.
- Wrote `research_workspace/notes/MODEL_REDESIGN_2026-03-23.md` as the current redesign note.

## Open Questions

- Whether the MM SAM-Adapter baseline should be re-run directly through the new reporting path, or only normalized from its saved JSON.
- Whether the next model should keep any form of deep cross-modal attention at `f3/f4`, or move fully to convolutional/local mixing.
- Which metric definition will be treated as the paper-facing official number on FMB.

## Next Actions

- Re-evaluate the current best CACAF checkpoint with both absent-class policies and record the numbers.
- Re-evaluate the MM SAM-Adapter baseline under the same template.
- Implement the first redesign prototype: region-reliability fusion + detail-semantic decoupled decoder.
