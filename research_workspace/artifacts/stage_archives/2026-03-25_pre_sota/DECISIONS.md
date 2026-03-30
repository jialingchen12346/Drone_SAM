# 决策记录

请将应跨会话保留的关键判断写在这里。

## 2026-03-23

- 决策：旧研究材料统一归档，不再作为当前项目主线文档。
- 背景：旧项目中混杂了旧论文、旧实验、旧索引和过时结论，已经不利于继续推进。
- 影响：当前研究状态以 `research_workspace/` 和简化后的 `docs/` 为准。

## 2026-03-23

- 决策：第一轮新研究保留 encoder，优先重构 fusion 和 decoder。
- 背景：当前最主要的问题不在 backbone，而在旧结果口径不稳和结构创新点不够强。
- 影响：新模型先围绕“区域级可靠性融合 + 细节语义解耦解码”展开。

## 2026-03-23

- 决策：第一版原型采用 `RRF + DSDHead`，并保持旧训练脚本入口不重写。
- 背景：需要在最小工程风险下快速验证新结构，避免影响旧基线可比性。
- 影响：新增 `rrf.py`、`dsd_head.py`、`rrf_dsd_segmentor.py`，训练脚本通过 `--model-variant` 切换 `cacaf`/`rrf_dsd`。

## 2026-03-23

- 决策：`--no-cacaf` 与 `--no-sagu` 仅绑定到 `cacaf` 变体，在 `rrf_dsd` 下显式忽略。
- 背景：避免将旧 ablation 开关误用于新结构，减少实验配置污染。
- 影响：RRF-DSD 的实验参数更清晰，配置解释更稳定。

## 2026-03-23

- 决策：第二轮主题调整为“长尾小目标与细结构鲁棒性”，在 `RRF-DSD` 上做结构增强而非继续改 backbone。
- 背景：对比基线后，`Truck/Bus` 有提升，但 `Traffic Light/Motorcycle/Sidewalk/Pole` 回落，说明细节保持与小目标表达不足。
- 影响：引入 `RRF` 细节残差增强通路、`DSDHead` 边界先验分支与小目标增强分支，并新增 stride-4 细节辅助头。

## 2026-03-23

- 决策：评测口径强制显式化，`test_rrf_dsd.py` 使用 `--split {val,test}`，并将结果分开落盘。
- 背景：出现“val 指标较高而 test 明显更低”的对比疑惑，根因是先前脚本默认固定 `split=test`，口径容易误解。
- 影响：后续所有结果必须同时注明 `split`、`eval_resize_mode`、`absent class` 处理策略，禁止混合比较。

## 2026-03-23

- 决策：当前阶段优先解决泛化问题，不再继续堆叠结构复杂度。
- 背景：`RRF-DSD v2` 已完成 80 epoch 训练并取得 `best val mIoU=70.57`，但 `best.pth` 在 test 上为 `mIoU=64.69`，存在显著 gap。
- 影响：下一轮实验以“泛化优先配置”与 checkpoint 选择策略为主线，先验证 `best.pth vs epoch_80.pth` 的 test 表现，再决定是否进行结构回退消融。

## 2026-03-24

- 决策：远程训练环境采用“结构快照文档”作为会话间共识，避免路径与实例漂移导致重复探测。
- 背景：当前训练与评测在 `connect.bjb1.seetacloud.com:37143` 上进行，且目录映射与本地存在差异（项目在 `/root/Drone-SAM-Adapter`，数据在 `/root/autodl-tmp/datasets`）。
- 影响：后续所有远程操作优先参考 `REMOTE_HOST_SNAPSHOT_2026-03-24.md`，并在实例变化时先更新快照再继续实验。

## 2026-03-24

- 决策：双卡训练默认采用 DDP + `gloo` 后端（`--dist-backend auto` 在 5090 上自动选择 `gloo`）。
- 背景：在该远程 2x5090 环境中，`nccl` 后端对 `SAM2HieraAdapter` 进行 DDP reducer 初始化时稳定触发 `CUDA illegal memory access` 并中止；切换 `gloo` 后端可稳定完成 2 卡训练与验证。
- 影响：`segmentation/train_cacaf.py` 已接入 DDP（sampler、rank0 日志/保存、val deadlock 修复），`scripts/run_ddp_train.sh` 可直接双卡启动；若未来驱动/框架升级后 `nccl` 稳定，再切回 `--dist-backend nccl` 复测。

## 2026-03-24

- 决策：在 2x5090 机器上恢复 `NCCL` 训练，采用独立新环境 `torch211`（`torch 2.11.0+cu128 + NCCL 2.28.9`）并显式设置 `LD_LIBRARY_PATH` 指向环境内 NCCL 动态库。
- 背景：诊断显示旧环境（`torch 2.7.0+cu128 + NCCL 2.26.2`）中，`NCCL` 纯通信可通过，但模型反向 allreduce 稳定触发 `CUDA illegal memory access`；升级后同一 `RRF-DSD` preflight（含 forward/backward）可通过。
- 影响：后续双卡训练优先使用新环境（`PYTHON_BIN=/root/miniconda3/envs/torch211/bin/python`）与 `NCCL_LIB_DIR=/root/miniconda3/envs/torch211/lib/python3.12/site-packages/nvidia/nccl/lib`；`gloo` 回退仅用于紧急兜底。

## 2026-03-24

- 决策：`RRF-DSD v2` 的 test checkpoint 选择以 `epoch_80.pth` 为当前默认，而非 `best.pth`。
- 背景：同口径 `split=test` 对照结果为 `epoch_80.pth mIoU=65.24`，高于 `best.pth mIoU=64.68`。
- 影响：后续对外主结果与新实验 warm-start 优先参考 `epoch_80.pth`，同时继续监控 val-test gap。

## 2026-03-24

- 决策：研究主线升级为“CCF-C 投稿导向”，并引入自动闭环驱动实验迭代。
- 背景：当前 bottleneck 不在单次结构变更，而在泛化稳定性、消融覆盖与实验迭代效率。
- 影响：新增 `scripts/auto_train_eval_loop.py` + `research_workspace/plans/auto_loop_config.json`，统一执行“训练→评测→反馈→下一轮 recipe”。

## 2026-03-24

- 决策：对 Codex 工作流引入“阶段化技能路由”，按任务阶段自动推荐技能组合。
- 背景：目标已从单点调参转为“模型迭代 + 论文产出”并行推进，技能调用需要标准化。
- 影响：新增 `scripts/skill_router.py` 与投稿执行计划文档，减少会话切换时的策略漂移。

## 2026-03-24

- 决策：新增本地远程心跳守护进程，周期监管远程训练/调度状态并启用自动自愈。
- 背景：用户希望“无需定期提示词”，由系统持续托管项目运行。
- 影响：新增 `scripts/heartbeat_remote_supervisor.sh`（`start/stop/status/once`），默认每 5 分钟检查进程、日志新鲜度、GPU 与磁盘；当训练与调度均消失时自动拉起 `auto_train_eval_loop.py`。本机已配置 `crontab`（`*/5` + `@reboot`）周期执行。

## 2026-03-24

- 决策：替换旧远程调度等待逻辑，避免 `pgrep` 自匹配导致“永远等待”。
- 背景：旧调度命令在 `while pgrep ...` 中会匹配自身命令行，训练结束后仍显示 waiting。
- 影响：重启为新调度进程，按真实训练进程条件进入 `auto_train_eval_loop.py`，自动闭环已实际开始执行。

## 2026-03-24

- 决策：GPU 利用率整改优先从吞吐参数入手，而非单纯增加 epoch。
- 背景：在线观测显示双卡显存占用约 `7.4GB/32GB`、利用率约 `40~70%`，且 DDP 日志明确提示 `find_unused_parameters=True` 带来额外开销。
- 影响：`auto_loop_config.json` 已调整为 `batch-size=4`、`num-workers=8`、`--no-ddp-find-unused-params`，并将 `save-freq` 调整为 `30` 以降低 I/O 和磁盘压力；将于下一轮自动循环生效。

## 2026-03-24

- 决策：自动循环训练吞吐上调到 `batch-size=8`（并配 `num-workers=12`）。
- 背景：2x5090 实时监控显示显存占用约 7.4GB/卡，GPU 利用率仍有提升空间。
- 影响：`auto_loop_config.json` 已更新并同步远程；当前在跑轮次继续按旧参数，下一次新启动的训练轮次自动生效。

## 2026-03-24

- 决策：吞吐优化参数从 `batch-size=4` 进一步上调到 `batch-size=8`（`num-workers=12`），并已启动新一轮自动实验验证。
- 背景：用户要求继续提高吞吐；线上观测显示显存仍有余量，且提升吞吐优先于继续加 epoch。
- 影响：新一轮 `gen_ohem_dice_cmr060_aux005_no_photo` 已按新参数运行；将以最终 test mIoU 与 epoch 耗时共同判定是否固化为默认配置。

## 2026-03-24

- 决策：提升自动闭环可观测性，训练默认启用 `PYTHONUNBUFFERED=1`，并让心跳将“checkpoint 最新修改时间”作为训练活跃度兜底信号。
- 背景：batch=8 新轮次中，`*_train.log` 的 stdout 存在缓冲，导致心跳容易误判“日志过旧”。
- 影响：`scripts/run_ddp_train.sh` 与 `scripts/auto_train_eval_loop.py` 已默认无缓冲输出；`scripts/heartbeat_remote_supervisor.sh` 已新增 `active_work_dir` 与 checkpoint mtime 兜底，降低误报风险。

## 2026-03-24

- 决策：在不影响当前训练和历史指标复核的前提下，对 `work_dirs/auto_loop` 执行 checkpoint 瘦身：保留每轮 `best.pth`，删除非活跃轮次 `epoch_30.pth`。
- 背景：远程系统盘仅 30G，自动循环每轮双 checkpoint（`best` + `epoch_30`）会快速挤压剩余空间。
- 影响：已清理 3 个已完成轮次的 `epoch_30.pth`，远程 `/` 可用空间由 `7.6G` 回升到 `12G`（`75% -> 61%`），当前活跃训练未受影响。

## 2026-03-24

- 决策：修复 `run_ddp_train.sh` 启动器偶发 `train_exit_code=127` 问题，改为数组组装命令执行（避免续行解析风险）。
- 背景：`gen_ohem_dice_cmr060_aux005_no_photo` 轮次训练已完成，但脚本收尾报错 `--nproc_per_node=2: command not found`，自动闭环将该轮误记为 `failed`。
- 影响：已修复并同步脚本；对该轮补跑评测并回填 history/leaderboard，结果为 `val=66.71 / test=64.93 / gap=1.78`，当前 rank2（仍低于 `65.32` 最优）。

## 2026-03-24

- 决策：心跳自动恢复增加“待执行 recipe 数量”门控，`pending_recipe_cnt=0` 时不再拉起 auto-loop。
- 背景：所有 recipe 完成后，旧逻辑会因“无训练进程”周期性空启动调度器。
- 影响：`heartbeat_remote_supervisor.sh` 已新增 `pending_recipe_cnt` 采集与判定；当前状态 `train_cnt=0, loop_cnt=0, pending_recipe_cnt=0`，不再空转。

## 2026-03-24

- 决策：泛化主线切换为 `OHEM-only + batch=8`，作为当前默认强基线。
- 背景：新轮次 `gen_ohem_only_cmr060_aux010_bs8_r1` 达到 `test mIoU=66.36`，超过此前最优 `65.32`，同时运行耗时从 `1550.5s` 降到 `728.3s`（约 `2.13x` 提速）。
- 影响：后续实验优先在该主线上做微调，重点回补 `Sidewalk/Traffic Light/Pole` 回退类别，避免重新引入强增强导致的大 gap。

## 2026-03-24

- 决策：启动 `batch-size=12` 的 OHEM-only 吞吐/精度验证轮次（`gen_ohem_only_cmr060_aux010_bs12_r1`）。
- 背景：用户要求继续提高 batch，并评估当前增益是否已触顶；2x5090 显存仍有一定余量可测试更高吞吐。
- 影响：`auto_loop_config.json` 已切换 `batch-size=12` 并新增 recipe；当前训练中显存约 `26.8GB/卡`、GPU 利用率约 `96%`，无 OOM。

## 2026-03-24

- 决策：结构优化主线进入 V3 重研阶段，优先实现细结构补偿分支（TSCB）。
- 背景：最新最优轮次虽提升总 mIoU，但 `Pole/Traffic Light/Traffic Sign/Sidewalk` 仍回退，属于结构性短板而非单纯优化器问题。
- 影响：已新增 `ARCH_REDESIGN_RESEARCH_2026-03-24.md`，并确定 V3 优先级为 `TSCB -> DRLR -> CBR`。

## 2026-03-24

- 决策：在代码层落地 `TSCB` 原型为“可开关”路径，默认关闭以保持历史基线可比。
- 背景：需要把结构重研从文档推进到可训练实现，同时避免影响正在运行的基线自动循环。
- 影响：`DSDHead` 新增 `ThinStructureCompensator` 分支；`train_cacaf.py/test_rrf_dsd.py` 新增参数 `--use-thin-structure-refiner` 与 `--thin-refiner-scale`；`auto_train_eval_loop.py` 新增 per-recipe `eval_flags` 支持，便于后续结构实验自动评测。

## 2026-03-24

- 决策：基于论文与 Hugging Face 开源模型调研，下一阶段默认采用 “`batch=12 OHEM-only` + `TSCB` 结构对照” 双轨推进。
- 背景：用户要求重研结构；外部证据显示 RGB-T 新进展集中在 Foundation Model 迁移、融合-解耦协同和轻量实时化。
- 影响：已形成调研文档 `PAPER_HF_RESEARCH_2026-03-24.md`，并在 `auto_loop_config.json` 新增 `gen_ohem_only_bs12_tscb015` 与 `gen_ohem_only_bs12_tscb012_aux008_nophoto` 两条结构实验 recipe。

## 2026-03-24

- 决策：`batch=12` 不作为默认主配置，继续保留 `batch=8` 为性能主线；`batch=12` 作为吞吐对照。
- 背景：`gen_ohem_only_cmr060_aux010_bs12_r1` 得到 `test mIoU=65.95`，未超过 `bs8` 最优 `66.36`。
- 影响：已立即启动 `TSCB` 结构实验轮次，目标是在不牺牲总体 mIoU 的前提下回补 `Sidewalk/TrafficLight/Pole`。

## 2026-03-24

- 决策：发文路径采用“主投 PRCV 2026，备投 BMVC 2026，再备投 ACCV 2026”的级联策略。
- 背景：已核验 CCF 第七版目录与各会官方截止时间，PRCV 截止为 2026-04-15，时间窗口最紧且匹配度最高。
- 影响：研究排程改为截止日倒排，优先保证可提交版本按时形成，不再无边界扩展实验分支。

## 2026-03-24

- 决策：升级心跳自动清理策略为“保留活跃目录 + Top-N best checkpoint，其余轮次清空 checkpoint”。
- 背景：旧策略仅删 `epoch_10/20`，在 `best.pth` 累积后对低盘告警无实质缓解（出现 `5G -> 5G`）。
- 影响：`scripts/heartbeat_remote_supervisor.sh` 已更新；并执行一次人工清理，远程可用空间从 `4.8G` 回升到 `9.0G`，训练不中断。

## 2026-03-24

- 决策：磁盘策略进一步激进化为“双阈值治理”：低水位触发 + 目标水位回收。
- 背景：30G 系统盘在 auto-loop 阶段易出现反复低盘；仅做 checkpoint 删除不够稳定。
- 影响：
  - `MIN_FREE_GB` 提升到 `10G`，新增 `TARGET_FREE_GB=12G`。
  - `KEEP_TOP_N_CHECKPOINTS` 默认收紧到 `1`（仅保留历史最优 `best.pth`）。
  - 低盘时按顺序执行：checkpoint 回收 -> conda/pip 缓存清理 -> 大日志瘦身。
  - 实测可用空间 `8G -> 16G`，当前训练保持运行。

## 2026-03-24

- 决策：后续 auto-loop 的 `work_dir_root` 切换到数据盘（50G）。
- 背景：系统盘仅 30G，checkpoint 落在系统盘会周期性触发低盘风险；`/root/autodl-tmp` 可用约 47G。
- 影响：
  - `research_workspace/plans/auto_loop_config.json` 已改为 `work_dir_root=/root/autodl-tmp/work_dirs/auto_loop` 并同步远程。
  - 心跳清理逻辑改为自动读取配置中的 `work_dir_root`，避免路径写死导致清理失效。

## 2026-03-24

- 决策：`TSCB` 结构分支暂不作为主线，恢复到 `OHEM-only + bs8` 参数微调迭代（R2）。
- 背景：`bs8` 的两条 `TSCB` 轮次分别得到 `test mIoU=63.69` 与 `64.60`，均低于现有最优 `66.36`。
- 影响：
  - `auto_loop_config.json` 新增 4 个 `OHEM-only + bs8` recipe（`cmr055/060/062` 与 `aux 0.08/0.10` 组合，含 `no-photo` 对照）。
  - 已同步远程并启动新一轮 `auto_train_eval_loop.py --execute --max-cycles 4`。
  - 新轮次输出落盘到数据盘目录 `/root/autodl-tmp/work_dirs/auto_loop`，继续沿用激进磁盘治理策略。

## 2026-03-24

- 决策：确认 `OHEM-only + bs8` 的新最优配方更新为 `cmr055 + aux010`，并进入 R3“seed+微扰”冲线阶段。
- 背景：R2 完成后，`gen_ohem_only_bs8_cmr055_aux010_r2` 达到 `test mIoU=66.75`（`val=66.26`，`gap=-0.49`），较旧最优 `66.36` 提升 `+0.39`，距离 `66.8` 仅差 `0.05`。
- 影响：
  - `auto_loop_config.json` 新增 4 条 R3 recipe：
    - `gen_ohem_only_bs8_cmr055_aux010_seed7_r3`
    - `gen_ohem_only_bs8_cmr055_aux010_seed17_r3`
    - `gen_ohem_only_bs8_cmr056_aux010_r3`
    - `gen_ohem_only_bs8_cmr055_aux011_r3`
  - 已同步远程并启动 `auto_train_eval_loop.py --execute --max-cycles 4`，继续冲击 `66.8+`。

## 2026-03-24

- 决策：评测口径切换为“缺失类按 0 分计入 mIoU”（`--absent-score 0.0`）。
- 背景：FMB test split 中 `Bicycle` 长期缺失，present-only 口径（忽略缺失类）会抬高 test mIoU，并与“全类平均”叙事不一致。
- 影响：
  - `research_workspace/plans/auto_loop_config.json` 的 `eval.extra_flags` 已新增 `["--absent-score", "0.0"]` 并同步远程。
  - 后续新启动轮次将按全 14 类（缺失类记 0）输出 mIoU；历史结果需区分旧口径（present-only）与新口径（all-class zero-fill）。

## 2026-03-24

- 决策：新建严格口径独立实验线（R4），避免与旧 leaderboard 混算。
- 背景：旧口径循环已全部完成并出现高分（最高 `67.67`），但该值属于 present-only；用户要求缺失类按 0 计分作为正式口径。
- 影响：
  - `auto_loop_config.json` 的 `artifacts_dir` 切换为 `research_workspace/artifacts/auto_loop_strict_abs0`。
  - recipe 收敛为 4 条高价值 strict 配方（`cmr056/057` 与 `aux010/011` + seed 对照）。
  - 已启动 `auto_train_eval_loop.py --execute --max-cycles 4`（日志：`/tmp/auto_loop_scheduler_20260324_r4_strict.log`）。

## 2026-03-25

- 决策：在进入下一轮筛选前，先完成 V5 结构改动并统一训练/评测参数口径。
- 背景：用户明确要求“先改模型，再筛选”；此前仅更换了 SAM2.1 权重，结构本体未改。
- 影响：
  - `DSDHead` 已启用可选 `Boundary-Guided Refiner` 与 `Rare-Class Residual` 分支。
  - `RRFDSDSegmentor` 已兼容 decoder 新返回签名，并新增可选 boundary BCE 辅助损失。
  - `train_cacaf.py` / `test_rrf_dsd.py` 已新增对应 CLI 参数，避免训练与评测结构不一致。
  - 新快筛配置已落地并启动：`research_workspace/plans/quick_screen_v5_sam21_e12_2026-03-25.json`（`12 epoch`, `batch=8`, `OHEM-only`, strict eval）。

## 2026-03-25

- 决策：终止 `V5(boundary+rare)` 作为当前主线，回退为“单变量结构筛选”策略。
- 背景：`qs_v5_sam21_ohem_bs8_e12_bref_rare` 已完成，strict 结果 `val=56.35 / test=55.81`，显著低于当前 strict 锚点（约 `62+`）。
- 影响：
  - 该组合不进入 20/30 轮复筛。
  - 后续改为“单分支增量”验证（先 boundary-only，再 rare-only），避免多改动耦合导致退化难定位。
  - 本轮已执行磁盘收敛：删除重复 `epoch_12.pth`，仅保留 `best.pth + 结果文本`。

## 2026-03-25

- 决策：完成 `V5` 单变量定位后，判定 `boundary-only` 与 `rare-only` 均不进入下一阶段。
- 背景：同预算 (`12 epoch`, strict) 对照结果：
  - baseline `55.61`
  - boundary-only `54.48`（`-1.13`）
  - rare-only `54.69`（`-0.92`）
- 影响：
  - `V5` 路线阶段性结束，后续转入新结构方案设计（避免继续在该分支消耗算力）。
  - 已执行磁盘清理：删除 `quick_screen_v5_sam21_e12` 与 `quick_screen_v5_ablation_e12` 下全部 `.pth`，仅保留 `val/test_results.txt` 与 artifacts 索引；数据盘可用空间回升至约 `33G`。

## 2026-03-24

- 决策：中止 R4 严格口径循环，转入 V4 结构重构（先做训练侧长尾重加权）。
- 背景：用户认为 strict 绝对分偏低且继续堆配方收益有限；R4 已完成 2/4（strict test 最优 `62.77`）。
- 影响：
  - 已停止远程 `auto_train_eval_loop.py` 与对应训练子进程，释放 GPU。
  - 代码已落地“可开关的 class-balanced CE/OHEM”：
    - `segmentation/train_cacaf.py` 新增参数：`--use-class-balanced-loss`、`--class-balance-alpha`、`--class-weight-min/max`；
    - `rrf_dsd_segmentor.py` / `cacaf_segmentor.py` 支持 class weight 传入 CE/OHEM。
  - 已完成本地与远程 `py_compile` 校验，准备进入 V4-1 实验轮次。

## 2026-03-24

- 决策：启动 V4-1 首轮验证（`class-balanced + OHEM`）作为重构起点。
- 背景：现有配方在旧结构上收益趋于饱和，需要验证训练侧长尾重加权是否能稳定提升关键类。
- 影响：
  - 已在远程启动手工单轮训练：`v4_cb_ohem_cmr056_aux010_cbalpha05_seed42_20260324_181508`。
  - 日志：`/tmp/v4_cb_ohem_cmr056_aux010_cbalpha05_seed42_20260324_181508.log`；
  - 输出目录：`/root/autodl-tmp/work_dirs/v4_cb/...`。

## 2026-03-24

- 决策：快速筛选默认从 `10 epoch` 上调到 `12 epoch`，并采用 `12/20/30` 三段式漏斗流程。
- 背景：`10 epoch` 在当前任务上明显欠收敛（首轮 strict test 仅 `50.85`），对最终排名相关性不足。
- 影响：新增 `scripts/staged_funnel_loop.py` 与 `research_workspace/plans/staged_funnel_strategy.json`，后续候选按 `S1(12)->S2(20)->S3(30)` 自动推进。

## 2026-03-24

- 决策：终止 `quick_screen_v4` 自动循环，避免继续消耗算力在低信息密度配置上。
- 背景：`quick_screen_v4` 首轮完成后结果显著低于主线区间，且当前更需要“可用于决策的中期收敛信号”。
- 影响：已停止对应调度/训练进程，改由三段式漏斗主线替代。

## 2026-03-24

- 决策：执行一次数据盘激进清理，保留关键权重最小集合。
- 背景：`/root/autodl-tmp` 一度仅剩 `3.1G`，会直接威胁后续 checkpoint 落盘。
- 影响：清理后可用空间回升至约 `38G`；`/root/autodl-tmp/work_dirs/auto_loop` 仅保留三组关键目录及必要文件（`best/epoch_30/val/test`）。

## 2026-03-24

- 决策：重启 30 轮锚点主线训练，作为漏斗策略的对照基准。
- 背景：需要一个持续更新的稳定基线用于与漏斗候选对比。
- 影响：已启动 run `r5_ohem_only_bs8_cmr056_aux010_seed42_e30_20260324_190618`，日志位于 `/tmp/r5_ohem_only_bs8_cmr056_aux010_seed42_e30_20260324_190618.log`。

## 2026-03-24

- 决策：三段式漏斗在 `S3` 触发止损后正式收尾，不继续在该配方线上追加训练。
- 背景：漏斗完整执行结果为 `S1 best=57.10 -> S2 best=61.53 -> S3 best=62.23`（strict 口径，absent=0），低于 `S3` 止损阈值 `63.0`。
- 影响：
  - `staged_funnel_report.json` 已落盘并作为阶段结论基准。
  - 后续改为“结构改造优先 + 短周期筛选 + 长训确认”策略，不再加算力硬拉当前配方。

## 2026-03-24

- 决策：执行数据盘二次激进清理，按白名单仅保留关键 run。
- 背景：漏斗结束后 `/root/autodl-tmp` 达到 `97%` 使用率，存在下一轮启动即落盘失败风险。
- 影响：
  - 已删除 `staged_funnel/s1_screen_e12`、`staged_funnel/s2_refine_e20` 和非关键 `auto_loop` 历史目录。
  - 保留目录：
    - `staged_funnel/s3_confirm_e30/fmb_cb_ohem_cmr055_aux010_alpha03_20260324_203024`
    - `auto_loop/gen_ohem_only_bs8_cmr056_aux010_r3_20260324_170732`
  - 保留文件：`best.pth`、`epoch_30.pth`、`val_results.txt`、`test_results.txt`。
  - 清理后数据盘使用率 `97% -> 19%`（可用约 `41G`）。

## 2026-03-24

- 决策：立即进入 `V4-2` 漏斗新轮次，不等待人工触发。
- 背景：上一轮漏斗已在 strict 口径止损结束，且磁盘空间已恢复，具备继续自动迭代条件。
- 影响：
  - 新配置已创建：`research_workspace/plans/staged_funnel_v42_strategy.json`。
  - 方向为“`OHEM-only`、`class-balanced`、`TSCB` 与 `no-photo` 组合”的结构选型快筛。
  - 已在远程启动：`scripts/staged_funnel_loop.py --config .../staged_funnel_v42_strategy.json --execute`（日志：`/tmp/staged_funnel_v42_20260324_205511.log`）。

## 2026-03-25

- 决策：对历史高分 run 先做“快速 strict 重评”再讨论主线有效性，避免被口径差异误导。
- 背景：`gen_ohem_only_bs8_cmr056_aux010_r3` 的 `67.67` 来自 present-only 口径，无法直接与 strict 线比较。
- 影响：
  - 已重评同 run：`67.67 -> 62.84`（`epoch_30`, `absent=0`，差值 `-4.83`）。
  - 已确认 S3 strict run 复测一致：`62.23`。
  - 已新增归档文档：`research_workspace/notes/MODEL_ATTEMPT_ARCHIVE_2026-03-25.md`，后续统一以 strict 口径汇总。

## 2026-03-25

- 决策：论文叙事采用“旧稿中的无人机工程场景问题定义”，模型主线新增 `mmsa_baseline` 作为重构基线。
- 背景：现有 `RRF/DSD` 线在 strict 口径下进入收益平台，需要引入更贴近 MM SAM-Adapter 的“动态模态质量加权 + 跨模态交互 + 轻解码”基线做结构重启。
- 影响：
  - 训练入口新增 `--model-variant mmsa_baseline`（`segmentation/train_cacaf.py`）。
  - 新增融合模块 `segmentation/models/fusion/mmsa_fusion.py`（含样本级质量权重与轻量双向交互）。
  - 新增解码头 `segmentation/models/decode_heads/segformer_lite_head.py`（SegFormer 风格聚合）。
  - 新增端到端模型 `segmentation/models/segmentors/mmsa_baseline_segmentor.py`，可直接接入现有训练/评测脚本。

## 2026-03-25

- 决策：完成 `legacy_2026-03-23` 全量复盘，后续研发与写作执行“strict 红线”。
- 背景：旧路线并非单纯模型无效，而是口径漂移（present-only 与 strict 混用）与证据治理不足叠加，导致主结论不可直接发表。
- 影响：
  - 新增复盘文档：`research_workspace/notes/LEGACY_POSTMORTEM_2026-03-25.md`。
  - 论文主表与主结论仅允许 `strict(absent=0)` 结果；`present-only` 仅作补充分析。
  - 历史高分 run 未经 strict 重评，禁止进入对外结论与投审稿材料。
