# Experiment Summary

Total runs: 5

| Run | Type | mIoU | mAcc | aAcc | Epoch | Path |
|---|---:|---:|---:|---:|---:|---|
| v5_ablation_no_ohem | log | 68.62 | 75.75 | 93.27 | 59 | `work_dirs/v5_ablation_no_ohem/eval_test_epoch60.log` |
| v5_ablation_full | log | 66.79 | 72.89 | 93.46 | 59 | `work_dirs/v5_ablation_full/eval_test_epoch60.log` |
| Segformer_MMSAM_adapter_large_FMB_800x800_ss_RGBTHERM | json | 66.1 | 72.09 | 93.95 |  | `segmentation/work_dirs/Segformer_MMSAM_adapter_large_FMB_800x800_ss_RGBTHERM/eval_single_scale_20260304_164206_FMB_RGBThermal_checkpoint.json` |
| v5_ablation_no_cacaf | log | 64.71 | 70.53 | 93.45 | 59 | `work_dirs/v5_ablation_no_cacaf/eval_test_epoch60.log` |
| v5_ablation_no_sagu | log | 64.67 | 70.57 | 93.27 | 59 | `work_dirs/v5_ablation_no_sagu/eval_test_epoch60.log` |
