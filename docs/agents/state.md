# 任务状态

## 当前状态

ALIGNN reliability classifier task ACTIVE. User requests lightweight classifier on final ALIGNN features and selected thresholds0.1/0.2/0.5eV. Freeze verified MP gap ALIGNN; extract256dim pre-fc pooled embeddings; construct correctness from DFT error. Plan train-meta data from original5000validation IDs, untouched4239test IDs, not original60000training errors. Check cached structures/labels on server before downloads. No new DFT.

## 当前计划

Implement experiments/mattergen/alignn_reliability/ module with README, dataset provenance/splits, frozen batch feature extraction, logistic and tiny MLP reliability heads, calibration and risk-coverage evaluation. Unit tests for labels/splits/features/reliability metrics; local tests+commit/push/bundle then SlurmGPUfeatures and CPUheads/audit. Explicit limits: original validation selection exposure, in-distribution equilibrium labels, no noisy/OOD correctness guarantee, no RL integration yet.

## 变更记录

- 2026-10-22 11:52:26 +08:00：开始 Guqq2 连接及网络设置同步；仅进行轻量 SSH 与网络诊断，不读取实验结果、不提交计算作业。缺少已确认 Git remote，不进行推测性 clone。

- 2026-10-22 11:55:16 +08:00：完成 Guqq2 网络同步验证与文档记录；仅关闭本任务代理，确认 1080 无监听。现有用户普通会话保持不变。

- 2026-10-22 12:20:36 +08:00：开始应用户请求核实 Guqq2 GPU 数量；仅执行轻量状态查询。


- 2026-10-22 12:23:59 +08:00：完成 GPU 清单和 Slurm 状态查询；确认单张 RTX 5090 当前空闲。

- 2026-10-30T16:25:26: Conditional trajectory implementation phase complete; ten tests and syntax checks passed. Enter synchronization/Slurm run stage. Cached native bandgap checkpoint and both environments confirmed; server idle at preflight. Native joint generation differs from fixed-composition CSP; MASK gaps remain unavailable.

- 2026-10-30T16:29:45: Additional stage-analysis implementation complete with seven tests. GPU505 observed healthy on node42; first conditional batch completed81.08s. Next submit CPU-only phase analysis after audit506.

- 2026-10-30T16:40:45: GPU/primary audit/phase jobs505/506/507 complete successfully. Enter final reporting QA: revise empty fixed-cohort panel and persist same-path force summary; rerun CPU only, verify raw hashes unchanged, then download final figures and document interpretation. No inference, training or relaxation rerun.

- 2026-10-30T16:47:30: Final artifact QA exposed a stale prior-stage SVG checksum; raw trajectories and predictions unchanged. Ownership regression now passes ten tests. Reenter CPU artifact integration; no GPU rerun needed.

- 2026-10-30T16:52:08: Conditional trajectory task COMPLETE. GPU505 sampling/evaluation and all required CPU audits complete; final CPU510 fixed manifest ownership and passed. Local final archive/raw/model hashes/figures/numeric documentation verified. No active jobs submitted by this task remain; no proxy holder was started. Deliver figures, data and stage explanation.

- 2026-10-30T17:03:27: Public checkpoint availability verification COMPLETE; requested direct download has no verified matching public target. No global nonexistence claim: weight-only opaque repositories unverified. Documented exact scope/evidence; retain existing cached weights. Further CSP property fine-tuning or clamped joint-model sampling would be separate work, not silently substituted.

- 2026-10-30T17:13:10: Training-design task COMPLETE. Concrete protocol and implementation/test boundaries recorded and source-checked. Next experimental unit would audit labeled polymorph coverage, then implement adapter/loader flow and200optimizer-stepSlurmGPUprofile before5000-step-scale pilot; these have not been run.

- 2026-10-30T17:49:38: Conditional-adapter explanation/documentation correction COMPLETE. Source/AST/dimension/parameter checks passed. Existing training proposal remains unlaunched; no new compute or experimental result changes.

- 2026-10-30T19:14:57: Template-local MatterGen editing design COMPLETE, source/grid/document checks passed. Protocol is proposed only, no new model or measured improvement. Existing bandgap checkpoint can be investigated for low-noise editing before training a dedicated CSP; species/locality and relaxed property gains remain validation requirements.

- 2026-10-31T15:57:25: Reliability implementation unit passed local8tests/staticchecks. Cached megnet ZIP and pinned ALIGNN weights found; serverGPU idle at preflight; reused Torch2.7.1cu128/sklearn1.7.2 environment verified. Enter source synchronization and CPUprepare->GPUfeatures. Report/CPUtraining integration follows; no DFT.

- 2026-10-31T16:03:46: Enter classifier training/report stage.511prepared3540/760/700/4239 partitions successfully;512GPUfeatures healthy. Reporting module tests passed. Next synchronize report source and submit dependent CPUheads/audit, then verify and download learned heads/results.
