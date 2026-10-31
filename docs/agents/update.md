
- 2026-10-22 11:54:29 +08:00：验证 Guqq2 免密与反向 SOCKS 网络成功，新增 network.md 使用说明及任务记录；未修改 SSH 配置、服务器源码、用户普通 SSH 会话或提交计算任务。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: Implemented conditional_trajectory module: native joint bandgap CFG2/CFG0 paired sampling, RNG-neutral stride5 capture, actual terminal output, separate ALIGNN/MatterSim posthoc inference, support/mask/geometry coverage, paired and fixed-cohort target MAE, raw audit and plot exports. Local tests pass; next synchronize and submit Slurm GPU plus dependent CPU audit.

- 2026-10-30T16:29:45: Added posthoc phase.py/phase.sbatch: event timing and same-path error changes, event-aligned gap curves, force/separation/formation-energy probes. Uses existing raw observations only, no change to running generation/evaluation505 or primary audit506.

- 2026-10-30T16:40:45: Completed native paired trajectory experiment: target2.5eV; terminal ALIGNN MAE CFG2=1.918877612 versus CFG0=2.159905745. Conditional first-evaluable-to-terminal error mean increases0.116983906; only2/12improve despite11/12force reductions. Early states predominantly MASK; first known median970conditional/957.5unconditional. Empty fixed750/900cohorts prompted figure replacement with all-parent first-versus-terminal scatter and coverage-aware zoom, preserving raw predictions.

- 2026-10-30T16:47:30: Fixed stale artifact manifests after sequential rerenders: primary report owns only its gap plots/CSV, phase owns its plots/JSON with own final manifest. Single phase Slurm job now refreshes primary audit first. No sampling/inference logic or cached results changed.

- 2026-10-30T16:52:08: COMPLETE requested bandgap-conditioned ALIGNN trajectory plots and stage interpretation. Final evidence conditional-gap-20261030-final; source8e381e0, generation89c3e3c. Conditional MAE1.801894atfirsteligible->1.918878terminal,2/12improve,11/12forcesreduce; control2.211269->2.159906,7/12improve,12/12forcesreduce. Terminal conditional11.16%lower average target error, descriptive smallsample only. Early MASK prevents gap evaluation; coverage and fixed-parent comparisons preserved. No relaxation/DFT/SUN added.

- 2026-10-30T17:03:27: Completed public bandgap+CSP checkpoint search requested for direct download. Official Microsoft HF catalog and dft_band_gap config, GSS release/config and12publicHFmattergenrepositories inspected; no verified eligible downloadable checkpoint found. Recorded scoped availability and alternatives in CSP README. No server connection, new large weight, training or sampling performed. Existing joint gap and pure CSP weights remain distinct.

- 2026-10-30T17:13:10: Completed adapter-first bandgap-conditioned CSP training design in csp/TRAINING_PLAN.md. Includes DFT label/polymorph coverage audit, FT group splits versus base pretraining contamination, fixed-species noising/native loss, frozen-adapter pilot then possible unfreezing, proposed hyperparameters, condition-loader fix, initialization/gradient/source tests, matched attainable-target evaluation before/after MatterSim. No training, data preprocessing or server jobs launched.

- 2026-10-30T17:49:38: Clarified exact MatterGen conditional adapter and corrected training proposal: default numeric-gap embedding is fixed sinusoidal, not a trainable embedding. Trainable parts are per-block MLP and zero-initialized mixing layers. Updated gradient checks to expect mix-layer learning first, later MLP learning, never parameter gradients on fixed encoder. Recorded frozen/full-FT unconditional distinction and4.1984Mcontrolparameter derivation.

- 2026-10-30T19:14:57: Added csp/TEMPLATE_EDITING.md proposal for near-final templates: low-noise forward geometry and partial reverse, immutable species, candidate relaxation and constrained best-of-N. Existing bandgap joint model proposed for low-noise CFG0/CFG2 comparison without new training, distinct from cold-start CSP. Identified sample() prior reset and discrete1000-grid trap; proposed fair direct/noise/CFG0/CFG2 controls and optional accepted-improvement iterations. No jobs launched.

- 2026-10-31T15:57:25: Implemented alignn_reliability core/prepare/features/head/train and README, with all three user-selected tolerances. Uses cached MEGNet dataset, original5000val for composition-grouped meta splits and original4239test untouched. Frozen256dim pre-fc features with same-forward reconstruction/nativecontrols; linear and16513paramMLP heads, separate calibration/riskcoverage metrics.8tests pass. Next submit CPU dataset selection and dependent GPUfeature extraction; finish report while features run.

- 2026-10-31T16:03:46: Added report.py and CPUhead-training/audit/archive wrapper. Reports calibrated selective risk/MAE and calibration curves, scalar-only baseline, three error tolerances, unseen-meta-composition/nonmetal subsets and cross-threshold monotonicity diagnostic. Independent original-ID exclusion/scaler/probability/AUC audit implemented.9tests pass.511complete,512inprogress; queue CPU heads after512.

- 2026-10-31T16:17:52: Classifier training complete:256->64->1MLP16513params, threshold0.1/0.2/0.5eV, feature inference309.18s and head training9.25s. Global performance positive but predicted-gap>0.1domain AUROC0.608/0.623/0.706; tau0.2p90only23accepted/17correct. Independent heads violate cross-threshold monotonicity25.5%, so no jointly calibrated CDF or GRPO reliability guarantee. Add targeted current-native/output discrepancy diagnostic; no retuning.

- 2026-10-31T16:33:41: Corrected read-only discrepancy audit to separate exact fractional-input batching from Cartesian roundtrip representation. Original514failure onmvc-9126 remains recorded; added exact-input forward/native controls, actual fractional delta and topology hashes, fresh per-attempt loader. Classifier outputs/selection unchanged.

- 2026-10-31T16:41:56: Completed ALIGNN correctness classifiers:256dim pre-fc frozen features, thresholds0.1/0.2/0.5eV, three baselines each, heldout4239test. Verified deployment ZIP246531B SHA0972fe94a4b01d725305bfb3687f3a2a3bef2a9097410b35beb8f499b27e244d plus plots/report. Diagnostics prove exact-input batch agreement but identify graph sensitivity at floating roundtrip; no retuning. User-facing conclusion emphasizes nonzero-gap limitation over aggregate gains.

- 2026-10-31T22:16:59: Implement exact-training-split bandgap statistics in existing alignn_reliability module: checkpoint-hash/split provenance,60000label selection, exact/rounded modes and bins, positive/zero partitions, target windows, compressed ID/label output. No predictor changes or GRPO training. Local15tests PASS; next sync verified source and submit CPU Slurm.
