
- 2026-10-22 11:54:29 +08:00：验证 Guqq2 免密与反向 SOCKS 网络成功，新增 network.md 使用说明及任务记录；未修改 SSH 配置、服务器源码、用户普通 SSH 会话或提交计算任务。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: Implemented conditional_trajectory module: native joint bandgap CFG2/CFG0 paired sampling, RNG-neutral stride5 capture, actual terminal output, separate ALIGNN/MatterSim posthoc inference, support/mask/geometry coverage, paired and fixed-cohort target MAE, raw audit and plot exports. Local tests pass; next synchronize and submit Slurm GPU plus dependent CPU audit.

- 2026-10-30T16:29:45: Added posthoc phase.py/phase.sbatch: event timing and same-path error changes, event-aligned gap curves, force/separation/formation-energy probes. Uses existing raw observations only, no change to running generation/evaluation505 or primary audit506.

- 2026-10-30T16:40:45: Completed native paired trajectory experiment: target2.5eV; terminal ALIGNN MAE CFG2=1.918877612 versus CFG0=2.159905745. Conditional first-evaluable-to-terminal error mean increases0.116983906; only2/12improve despite11/12force reductions. Early states predominantly MASK; first known median970conditional/957.5unconditional. Empty fixed750/900cohorts prompted figure replacement with all-parent first-versus-terminal scatter and coverage-aware zoom, preserving raw predictions.

- 2026-10-30T16:47:30: Fixed stale artifact manifests after sequential rerenders: primary report owns only its gap plots/CSV, phase owns its plots/JSON with own final manifest. Single phase Slurm job now refreshes primary audit first. No sampling/inference logic or cached results changed.

- 2026-10-30T16:52:08: COMPLETE requested bandgap-conditioned ALIGNN trajectory plots and stage interpretation. Final evidence conditional-gap-20261030-final; source8e381e0, generation89c3e3c. Conditional MAE1.801894atfirsteligible->1.918878terminal,2/12improve,11/12forcesreduce; control2.211269->2.159906,7/12improve,12/12forcesreduce. Terminal conditional11.16%lower average target error, descriptive smallsample only. Early MASK prevents gap evaluation; coverage and fixed-parent comparisons preserved. No relaxation/DFT/SUN added.

- 2026-10-30T17:03:27: Completed public bandgap+CSP checkpoint search requested for direct download. Official Microsoft HF catalog and dft_band_gap config, GSS release/config and12publicHFmattergenrepositories inspected; no verified eligible downloadable checkpoint found. Recorded scoped availability and alternatives in CSP README. No server connection, new large weight, training or sampling performed. Existing joint gap and pure CSP weights remain distinct.
