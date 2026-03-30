# 会话交接

## 当前目标

- 清理旧研究后，启动新一轮 RGB-T 语义分割模型研究。
- 当前进入第二阶段：在已完成原型训练基础上，优先缩小 `val-test` 泛化差距。

## 本轮已完成

- 阶段性总结已沉淀：
  - `research_workspace/notes/STAGE_SUMMARY_2026-03-24_1350.md`

- 旧论文、旧实验、旧索引、旧计划和旧脚本已归档到 `archive/legacy_2026-03-23/`。
- 工作流文档已切换为中文。
- 当前最关键的调研结论已整理到 `ARCHITECTURE_RESEARCH_2026-03-23.md`。
- `RRF-DSD` 第一版骨架已完成：
  - `segmentation/models/fusion/rrf.py`
  - `segmentation/models/decode_heads/dsd_head.py`
  - `segmentation/models/segmentors/rrf_dsd_segmentor.py`
- 基于首轮反馈完成第二轮结构增强（面向小目标/细结构）：
  - `RRF` 增加细节残差增强通路（模态分歧细节 + 对齐特征 skip）
  - `DSDHead` 增加边界先验分支（stride-4）
  - `DSDHead` 增加小目标增强分支（f2-up + semantic-s4）
  - `DSDHead` 融合改为残差门控，避免细节流被语义门完全抑制
  - 新增 stride-4 细节辅助头
- 训练入口已接入模型切换：
  - `segmentation/train_cacaf.py` 新增 `--model-variant {cacaf,rrf_dsd}`（默认 `rrf_dsd`）
  - `--no-cacaf`、`--no-sagu` 仅在 `cacaf` 变体下生效
  - 新增 `--detail-aux-weight`（仅 `rrf_dsd` 使用）
- 已完成 `RRF-DSD v2` 训练（80 epoch）：
  - `best val mIoU=70.57`
  - `last val mIoU=69.81`
  - 训练完成日志：`/tmp/train_rrf_dsd_v2.log`
- 已完成 `best.pth` 测试集评测：
  - `test mIoU=64.69 / mAcc=71.60 / aAcc=93.21`
  - `Bicycle` 在 test split 中 absent（`nan [absent]`）
- 已完成同口径 checkpoint 对照（`split=test`）：
  - `best.pth`: `mIoU=64.68`
  - `epoch_80.pth`: `mIoU=65.24`
  - 当前 test 选择策略：优先 `epoch_80.pth`
- 评测脚本口径已修正：
  - `segmentation/test_rrf_dsd.py` 新增 `--split {val,test}`
  - 支持分别输出 `val_results.txt` 与 `test_results.txt`，避免再混淆口径
- 远程主机文件结构已完成盘点并沉淀快照：
  - `research_workspace/notes/REMOTE_HOST_SNAPSHOT_2026-03-24.md`
- 双卡 `NCCL` 兼容问题已完成定位与修复方案验证：
  - 旧环境（`torch 2.7.0+cu128 / NCCL 2.26.2`）在模型 backward allreduce 阶段稳定触发 `CUDA illegal memory access`
  - 新环境（`torch211`: `torch 2.11.0+cu128 / NCCL 2.28.9`）已通过 `ddp_model_preflight.py` 的 full 模式（forward/backward）
  - `scripts/run_ddp_train.sh` 已支持 `NCCL_LIB_DIR`，可在新环境下直接走 `DIST_BACKEND=nccl`
- 已启动 30 epoch 泛化优先双卡训练（`torch211 + NCCL`）：
  - 工作目录：`/root/Drone-SAM-Adapter/work_dirs/rrf_dsd_v2_gen30_nccl_ohemdice`
  - 当前已验证训练持续推进并可正常落盘 checkpoint
- 已完成远程磁盘应急清理（激进策略）：
  - 系统盘占用从 `95%` 降至 `47%`
  - 保留主实验关键权重与当前训练目录
- 已落地自动化闭环与技能路由工具：
  - `scripts/auto_train_eval_loop.py`
  - `research_workspace/plans/auto_loop_config.json`
  - `scripts/skill_router.py`
  - `research_workspace/plans/CCFC_PAPER_EXECUTION_PLAN_2026-03-24.md`
  - 已完成 `auto_train_eval_loop.py --dry-run` 与 `skill_router.py` 参数验证
- 已落地本地远程心跳守护：
  - `scripts/heartbeat_remote_supervisor.sh`
  - 默认每 5 分钟检查远程：训练/调度进程、日志新鲜度、GPU 利用率、磁盘可用空间、history 行数
  - 训练与调度均消失时自动自愈拉起 `auto_train_eval_loop.py`
  - 已配置本机 `crontab`：
    - `*/5 * * * * ... heartbeat_remote_supervisor.sh once`
    - `@reboot sleep 45 && ... heartbeat_remote_supervisor.sh once`
  - 已修复旧自动调度器“等待条件自匹配”问题，现由新调度进程直接执行 `auto_train_eval_loop.py`
  - 已修复“低盘不自动清理”问题：心跳脚本新增 `AUTO_DISK_CLEANUP`，低于阈值自动执行保守清理
  - 已增强“活跃训练判定”逻辑：新增 `active_work_dir` 采集，并将活跃目录中 checkpoint 的最新 mtime 作为日志 mtime 的兜底信号，减少 stdout 缓冲导致的 stale 误报

- 已完成 2026-03-24 13:55 磁盘二次瘦身（不影响当前训练）：
  - 清理策略：保留每轮 `best.pth`，删除已完成 auto-loop 轮次 `epoch_30.pth`
  - 清理结果：`/` 可用空间 `7.6G -> 12G`（`75% -> 61%`）
  - 当前活跃目录：`/root/Drone-SAM-Adapter/work_dirs/auto_loop/gen_ohem_dice_cmr060_aux005_no_photo_20260324_135002`
  - 训练/调度进程保持运行（`train_cnt>0`, `loop_cnt=1`）
- 已完成吞吐整改轮次 `gen_ohem_dice_cmr060_aux005_no_photo` 的结果回填：
  - 训练指标：`best val mIoU=66.67`（batch=8）
  - 补跑评测：`val=66.71 / test=64.93 / gap=1.78`
  - leaderboard 当前第 2（低于最优 `65.32`）
  - 原因说明：该轮训练后曾触发启动器 `exit_code=127`（已修复），自动循环误记为失败，现已补齐为 done 记录
- 当前自动循环状态（2026-03-24 14:06）：
  - `train_cnt=0`、`loop_cnt=0`、`pending_recipe_cnt=0`
  - 已无待执行 recipe，心跳已配置为不再空启动 auto-loop
- 新增完成轮次（2026-03-24 14:22）：
  - recipe: `gen_ohem_only_cmr060_aux010_bs8_r1`（OHEM-only, batch=8）
  - 结果：`test mIoU=66.36`（当前最佳），`val mIoU=65.85`，`gap=-0.51`
  - 与旧最优 `gen_ohem_only_cmr060_aux010` 对比：
    - `test +1.04`（`65.32 -> 66.36`）
    - 训练总耗时约 `2.13x` 提速（`1550.5s -> 728.3s`）
  - 目前无训练进程属于正常结束（`pending_recipe_cnt=0`），非故障停机

## 仍待解决

- 缩小 `val(70.57)` 与 `test(64.69)` 之间的泛化差距。
- 判断 `best.pth` 与 `epoch_80.pth` 在 test 上谁更优（模型选择策略）。
- 增加“泛化优先”训练配置（loss/采样/aux 权重）并做最小消融。
- 增加模态 masking（作为下一阶段，可开关，不并入当前主补丁）。
- 运行自动化闭环不少于 3 轮并输出 `leaderboard`。
- 形成 CCF-C 投稿实验包（主结果 + 消融 + 误差分析图表 + 复现说明）。

## 下一步动作

- 使用 `torch211 + NCCL` 启动正式双卡训练（替代旧 `gloo` 路径）：
  - `PYTHON_BIN=/root/miniconda3/envs/torch211/bin/python`
  - `NCCL_LIB_DIR=/root/miniconda3/envs/torch211/lib/python3.12/site-packages/nvidia/nccl/lib`
  - `DIST_BACKEND=nccl`
- 跑同口径评测对照：`best.pth` vs `epoch_80.pth` 在 `split=test` 的差异。
- 启动 30 epoch 泛化优先实验：
  - `--use-ohem --use-dice --dice-weight 0.5`
  - `--cat-max-ratio 0.6`
  - `--detail-aux-weight 0.1`
- 运行自动化闭环（先 dry-run，再 execute）：
  - `python scripts/auto_train_eval_loop.py --config research_workspace/plans/auto_loop_config.json --dry-run`
  - `python scripts/auto_train_eval_loop.py --config research_workspace/plans/auto_loop_config.json --execute --max-cycles 3`
- GPU 利用率整改（下一轮自动生效）：
  - `auto_loop_config.json` 已调整为 `batch-size=8`、`num-workers=12`
  - 已加 `--no-ddp-find-unused-params`（去除 DDP 无用参数额外图遍历）
  - `save-freq` 调整为 `30`，降低 I/O 与磁盘压力
  - 训练启动链路已默认 `PYTHONUNBUFFERED=1`，便于心跳按日志实时监管
- 按阶段自动匹配技能：
  - `python scripts/skill_router.py --stage analysis --val-miou 70.57 --test-miou 65.24 --json`
- 查看心跳状态：
  - `scripts/heartbeat_remote_supervisor.sh status`
  - `scripts/heartbeat_remote_supervisor.sh logs`
  - `scripts/heartbeat_remote_supervisor.sh alerts`
- 若 test mIoU 仍在 `64~65`，优先回退/弱化小目标增强分支后再做结构消融。
- 下一轮新增建议（因 batch=8 轮次未超越 65.32）：
  - 维持 `batch-size=8` 吞吐配置（已验证有效）
  - 在 `OHEM-only` 主线下优先回补 `Sidewalk/Traffic Light/Pole`：尝试 `no-photo-distort + detail_aux 0.08` 或 `cat-max-ratio 0.55`

## 最新状态（2026-03-24 14:34）

- 已按用户要求启动 `batch=12` 试验轮次：
  - recipe: `gen_ohem_only_cmr060_aux010_bs12_r1`
  - work_dir: `/root/Drone-SAM-Adapter/work_dirs/auto_loop/gen_ohem_only_cmr060_aux010_bs12_r1_20260324_143314`
  - 实时状态：`train_cnt=27`、`loop_cnt=1`、`gpu_util≈96/96`、显存约 `26.8GB/卡`
  - 当前无 OOM，训练正常推进
- 已完成结构重研文档：
  - `research_workspace/notes/ARCH_REDESIGN_RESEARCH_2026-03-24.md`
  - 结论：V3 优先级 `TSCB(细结构补偿)` -> `DRLR(双分辨率logit细化)` -> `CBR(类别边界重加权)`
- 已在代码中落地 TSCB 可开关原型（默认关闭，不影响旧基线）：
  - `segmentation/models/decode_heads/dsd_head.py`
  - `segmentation/models/segmentors/rrf_dsd_segmentor.py`
  - `segmentation/train_cacaf.py`（新增 `--use-thin-structure-refiner`, `--thin-refiner-scale`）
  - `segmentation/test_rrf_dsd.py`（同名评测参数）
  - `scripts/auto_train_eval_loop.py`（支持 recipe 级 `eval_flags`）
- 已完成论文与 HF 开源模型调研并形成可执行方案：
  - `research_workspace/notes/PAPER_HF_RESEARCH_2026-03-24.md`
  - 已在 `auto_loop_config.json` 增加后续结构实验 recipe：
    - `gen_ohem_only_bs12_tscb015`
    - `gen_ohem_only_bs12_tscb012_aux008_nophoto`

- `batch=12` 主线验证已完成（`gen_ohem_only_cmr060_aux010_bs12_r1`）：
  - `test mIoU=65.95`，低于 `bs8` 最优 `66.36`
  - 结论：`batch=12` 不固化为默认主配置，保留为吞吐对照

- 已启动结构实验（TSCB）：
  - 当前 auto-loop 单轮任务正在运行（preflight 后进入训练）
  - 目标：回补 `Sidewalk/Traffic Light/Pole` 并维持主指标稳定

- 吞吐提升新策略（2026-03-24）：
  - 自动循环基础参数已调整为 `batch-size=8`、`num-workers=12`
  - 当前进行中的第三轮维持旧参数（不中断），后续新启动轮次自动生效

## 最新状态（2026-03-24 15:10）

- 已完成“发文策略二次收敛”并升级文档：
  - `research_workspace/plans/PUBLICATION_STRATEGY_2026-03-24.md`（V2）
  - `research_workspace/notes/PUBLICATION_STRATEGY_EVIDENCE_2026-03-24.md`
- 已确认投稿级联与日期：
  - `PRCV 2026`：`2026-04-15`
  - `BMVC 2026`：`2026-05-22`
  - `ACCV 2026`：`2026-07-05`
- 远程训练仍在进行：
  - 活跃目录：`/root/Drone-SAM-Adapter/work_dirs/auto_loop/gen_ohem_only_bs12_tscb012_aux008_nophoto_20260324_150003`
  - 心跳采样：`train_cnt=3`、`loop_cnt=1`、`gpu_util=0,100`
- 已处理磁盘风险：
  - 手工清理已完成轮次 `epoch_30.pth`（保留 `best.pth`）
  - `/` 可用空间 `4.8G -> 9.0G`
- 已修复自动清理策略失效问题：
  - `scripts/heartbeat_remote_supervisor.sh` 现保留“活跃目录 + Top-N best”，其余轮次 checkpoint 自动清空

## 最新状态（2026-03-24 15:16）

- 磁盘策略已升级为“激进双阈值”并生效：
  - `MIN_FREE_GB=10`
  - `TARGET_FREE_GB=12`
  - `KEEP_TOP_N_CHECKPOINTS=1`
  - 低盘时追加执行 `conda clean -a`、`pip cache` 清理与大日志瘦身
- 自动清理实测结果：
  - `/` 可用空间由 `8G` 提升到 `16G`（`47%` 使用率）
  - 历史轮次多数目录已清空 checkpoint，仅保留当前最优 run 的 `best.pth`
- 训练状态：
  - 活跃目录：`/root/Drone-SAM-Adapter/work_dirs/auto_loop/gen_ohem_only_bs8_tscb012_aux008_nophoto_r1_20260324_151502`
  - 心跳采样：`train_cnt=27`、`loop_cnt=1`、`gpu_util≈96/92`

## 最新状态（2026-03-24 15:20）

- 已确认盘信息：
  - 系统盘：`/`，`30G`
  - 数据盘：`/root/autodl-tmp`，`50G`（可用约 `47G`）
- 已将后续自动实验输出目录切到数据盘：
  - `auto_loop_config.json` 中 `work_dir_root=/root/autodl-tmp/work_dirs/auto_loop`
  - 目录已创建并同步远程
- 心跳清理脚本已支持按配置读取 `work_dir_root`（不再写死系统盘路径）
- 当前正在进行的轮次因启动较早仍在系统盘目录；后续新轮次将落到数据盘

## 最新状态（2026-03-24 15:55）

- `TSCB` 两轮 `bs8` 结构实验已结束并回填：
  - `gen_ohem_only_bs8_tscb012_aux008_nophoto_r1`：`test mIoU=63.69`
  - `gen_ohem_only_bs8_tscb010_aux008_nophoto_r1`：`test mIoU=64.60`
  - 结论：当前 `TSCB` 配置未超过主线最优 `66.36`，先回到 `OHEM-only + bs8` 参数微调主线
- 已新增并同步 4 个 `OHEM-only + bs8` 二轮 recipe（R2）：
  - `gen_ohem_only_bs8_cmr055_aux008_nophoto_r2`
  - `gen_ohem_only_bs8_cmr055_aux010_r2`
  - `gen_ohem_only_bs8_cmr060_aux008_nophoto_r2`
  - `gen_ohem_only_bs8_cmr062_aux010_r2`
- 新一轮 auto-loop 已重启（`--max-cycles 4`）：
  - 调度日志：`/tmp/auto_loop_scheduler_20260324_r2.log`
  - 当前活跃目录：`/root/autodl-tmp/work_dirs/auto_loop/gen_ohem_only_bs8_cmr055_aux008_nophoto_r2_20260324_155226`
  - 运行状态：`GPU util≈84/89`，显存约 `19.6G/卡`

## 最新状态（2026-03-24 16:43）

- R2（4 轮）已全部完成（`14/14`）：
  - 新最优：`gen_ohem_only_bs8_cmr055_aux010_r2`
  - 指标：`test mIoU=66.75`，`val mIoU=66.26`，`gap=-0.49`
  - 相比旧最优 `66.36` 再提升 `+0.39`
- R2 其余新增结果：
  - `gen_ohem_only_bs8_cmr060_aux008_nophoto_r2`：`test=64.39`
  - `gen_ohem_only_bs8_cmr062_aux010_r2`：`test=66.07`
- 已启动 R3（seed/微扰冲线，目标 `66.8+`）：
  - 新增 recipe：`seed7`、`seed17`、`cmr056_aux010`、`cmr055_aux011`
  - 调度进程：`auto_train_eval_loop.py --max-cycles 4`
  - 当前活跃目录：`/root/autodl-tmp/work_dirs/auto_loop/gen_ohem_only_bs8_cmr055_aux010_seed7_r3_20260324_164259`

## 最新状态（2026-03-24 17:32）

- R3 已出现新高分轮次：
  - `gen_ohem_only_bs8_cmr056_aux010_r3`：`test mIoU=67.67`（旧口径：present-only）
  - `val mIoU=65.81`，`gap=-1.86`
- 用户确认评测口径应为“缺失类记 0”，已执行配置切换：
  - `auto_loop_config.json` 中评测统一追加 `--absent-score 0.0`
  - 该变更会影响后续新启动轮次；当前正在运行中的循环将于下一次重启后完全按新口径生效

## 最新状态（2026-03-24 17:42）

- 旧口径 R3 全部完成（`18/18`），当前最高为：
  - `gen_ohem_only_bs8_cmr056_aux010_r3`：`test mIoU=67.67`（present-only）
- 已切换到“严格口径实验线”（absent class = 0）并隔离 artifacts：
  - `artifacts_dir`: `research_workspace/artifacts/auto_loop_strict_abs0`
  - `eval.extra_flags`: `["--absent-score", "0.0"]`
- 已启动 R4（4 轮 strict 配方）：
  - `strict_bs8_cmr056_aux010_seed42_r4`
  - `strict_bs8_cmr056_aux010_seed7_r4`
  - `strict_bs8_cmr056_aux011_seed42_r4`
  - `strict_bs8_cmr057_aux010_seed42_r4`
  - 调度日志：`/tmp/auto_loop_scheduler_20260324_r4_strict.log`

## 最新状态（2026-03-24 18:12）

- R4 strict 循环已按用户意图中止（避免继续低收益消耗）：
  - 结束时进度：`2/4` 完成
  - strict 最优（all-class absent=0）：`test mIoU=62.77`
  - 相关进程已停，GPU 空闲
- 已进入 V4-1 代码改造（长尾类重加权）并同步远程：
  - `train_cacaf.py` 新增 class-balanced loss 开关与权重参数
  - `rrf_dsd_segmentor.py` / `cacaf_segmentor.py` 已支持 CE/OHEM class weight
  - 本地 + 远程语法检查通过（`py_compile`）
- V4-1 首轮训练已启动（手工单轮，非 auto-loop）：
  - run: `v4_cb_ohem_cmr056_aux010_cbalpha05_seed42_20260324_181508`
  - work_dir: `/root/autodl-tmp/work_dirs/v4_cb/v4_cb_ohem_cmr056_aux010_cbalpha05_seed42_20260324_181508`
  - log: `/tmp/v4_cb_ohem_cmr056_aux010_cbalpha05_seed42_20260324_181508.log`

## 最新状态（2026-03-24 18:55）

- 已完成对 V4-1 手工单轮的严格口径测试（`absent=0`）：
  - `test mIoU=62.53`
  - 未超过 strict 线当前参考最优 `62.77`，判定“不值得按同配置继续长训”
- 已按用户策略切换为“短周期选型快筛”（先选型再精调）：
  - 新配置：`research_workspace/plans/quick_screen_v4_config.json`
  - 统一严格口径评测：`--absent-score 0.0`
  - 训练周期缩短为 `10 epoch`，候选 recipe 共 `4` 条
  - 自动循环已启动：`/tmp/quick_screen_v4_scheduler_20260324.log`
  - 当前在跑第 1 条：`qs_base_ohem_cmr056_aux010_e10`

## 最新状态（2026-03-24 19:06）

- 已确认 `10 epoch` 快筛信号不足，策略改为 `12/20/30` 三段式漏斗。
- `quick_screen_v4` 已停止；避免继续低价值迭代。
- 远程数据盘已完成激进清理：
  - `/root/autodl-tmp` 可用空间 `3.1G -> 38G`
  - `auto_loop` 仅保留 3 个关键目录（含 `best.pth`/`epoch_30.pth`/`val/test` 结果）。
- 新增自动化脚本与配置：
  - `scripts/staged_funnel_loop.py`
  - `research_workspace/plans/staged_funnel_strategy.json`
  - `research_workspace/plans/STAGED_FUNNEL_PLAN_2026-03-24.md`
- 当前锚点训练已运行：
  - run: `r5_ohem_only_bs8_cmr056_aux010_seed42_e30_20260324_190618`
  - work_dir: `/root/autodl-tmp/work_dirs/auto_loop/r5_ohem_only_bs8_cmr056_aux010_seed42_e30_20260324_190618`
  - log: `/tmp/r5_ohem_only_bs8_cmr056_aux010_seed42_e30_20260324_190618.log`

## 下一步动作（交接后直接执行）

1. 持续监控当前 30 轮锚点，记录 `epoch 10/20/30` 指标与耗时。
2. 锚点完成后先跑 strict test，写入统一对照表。
3. 启动三段式漏斗：
   - `python scripts/staged_funnel_loop.py --config research_workspace/plans/staged_funnel_strategy.json --execute`
4. 将 S3 Top-1 作为论文主模型，S1/S2 结果归档为选型消融证据。

## 最新状态（2026-03-24 20:45）

- 三段式漏斗流程已确认全部结束，无训练进程残留：
  - `staged_funnel_loop.py` / `auto_train_eval_loop.py` / `train_cacaf.py` 均未运行
  - 双卡 GPU 处于空闲（`util=0%/0%`）
- 漏斗结果（strict，`--absent-score 0.0`）：
  - `S1 best test=57.10`（6 个候选）
  - `S2 best test=61.53`（Top-2 复筛）
  - `S3 best test=62.23`（Top-1 确认）
  - 结论：`S3` 触发止损（阈值 `63.0`），该轮策略收尾
- 报告与排行榜：
  - 总报告：`research_workspace/artifacts/staged_funnel/staged_funnel_report.json`
  - S3 榜单：`research_workspace/artifacts/staged_funnel/s3_confirm_e30/leaderboard.md`
- 数据盘二次应急清理已执行（激进白名单）：
  - 使用率：`97% -> 19%`（`avail ≈ 41G`）
  - 删除：`staged_funnel/s1_screen_e12`、`staged_funnel/s2_refine_e20`、非关键 `auto_loop` 历史目录
  - 保留关键目录：
    - `/root/autodl-tmp/work_dirs/staged_funnel/s3_confirm_e30/fmb_cb_ohem_cmr055_aux010_alpha03_20260324_203024`
    - `/root/autodl-tmp/work_dirs/auto_loop/gen_ohem_only_bs8_cmr056_aux010_r3_20260324_170732`
  - 以上目录均仅保留：`best.pth`、`epoch_30.pth`、`val_results.txt`、`test_results.txt`

## 下一步动作（更新）

1. 进入 `V4-2` 结构改造（优先处理 `Sidewalk/Traffic Light/Pole`），先做 `12 epoch` 快筛，不达阈值即止损。
2. 快筛通过后仅放大 Top-1 到 `30 epoch` 并按 strict 口径复评，避免再次全量长训。
3. 将 `67.67`（present-only）与 strict 口径结果并排整理为“口径差异说明”图表，作为论文实验设置说明的一部分。

## 最新状态（2026-03-24 20:55）

- 已直接启动 `V4-2` 三段式漏斗（新配置，服务于结构改造选型）：
  - 配置：`research_workspace/plans/staged_funnel_v42_strategy.json`
  - 后台日志：`/tmp/staged_funnel_v42_20260324_205511.log`
  - 进程：`staged_funnel_loop.py` + `auto_train_eval_loop.py` 已运行
- 当前正在跑 `S1` 第 1 个 recipe：
  - `v42_ohem_cmr055_aux010_seed42`
  - `work_dir=/root/autodl-tmp/work_dirs/staged_funnel_v42/s1_screen_e12/v42_ohem_cmr055_aux010_seed42_20260324_205511`
- 稳定性检查：
  - 双卡采样利用率约 `80~94%`（短采样窗口内有瞬时波动属正常）
  - 数据盘仍有 `~41G` 可用空间，可支撑本轮漏斗

## 最新状态（2026-03-25）

- 已按“简单快速重评”完成关键 checkpoint 的 strict 复核（`--absent-score 0.0`）：
  - `gen_ohem_only_bs8_cmr056_aux010_r3`：`67.67 (present-only) -> 62.84 (strict)`，差值 `-4.83`
  - `fmb_cb_ohem_cmr055_aux010_alpha03`（S3）：`62.23 (strict)`（复测一致）
- 重评结果已保存到远程对应目录：
  - `.../test_results_present_only_backup.txt`
  - `.../test_results_strict_abs0_epoch30_reval_20260325.txt`
  - `.../test_results_strict_abs0_reval_20260325.txt`
- 已完成本地归档文档：
  - `research_workspace/notes/MODEL_ATTEMPT_ARCHIVE_2026-03-25.md`

## 最新状态（2026-03-30）

- 研究主线已从“FMB 全监督卷分”切换为“标签效率导向（半监督 + 弱监督）”。
- 计划文档已更新：
  - research_workspace/plans/current_plan.yaml（重写为新里程碑与任务依赖）
  - research_workspace/plans/LABEL_EFFICIENT_RS_PLAN_2026-03-30.md（新增执行蓝图）
  - docs/RESEARCH_WORKFLOW.md（更新为新流程与阶段定义）
  - research_workspace/README.md（更新工作区说明与新原则）
- 口径红线保持不变：论文主结论仍以 strict(absent=0) 为准。
- 数据集策略调整为“FMB 短期主实验 + 遥感公开集补证据”。

## 下一步动作（交接后直接执行）

1. 完成协议脚本：
   - 固化 label ratio（1/2/5/10/20/50%）样本划分与随机种子。
   - 生成 point 弱标注并对齐评测映射。
2. 建立半监督训练最小闭环：
   - teacher-student + 伪标签刷新 + 一致性损失。
   - 先在 FMB 10% 标注预算上跑通。
3. 增加可靠性筛选：
   - 引入模态质量/不确定性权重。
   - 输出过滤前后伪标签质量对比。
4. 完成首轮门控评估：
   - 相对监督基线至少达到 +1.0 mIoU 或关键类均值 +1.5。
   - 连续 3 组无增益则回滚复杂模块。
5. 选定 1 个遥感数据集做补充实验准备（优先 LoveDA 或 UAVid）。
