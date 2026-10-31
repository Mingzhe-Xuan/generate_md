# 检查记录

## Guqq2 网络同步

- 2026-10-22 11:52:26 +08:00 计划：检查 ssh -G 的主机、用户、跳板机、密钥和 RemoteForward；BatchMode 验证公钥登录；ss 验证仅回环监听；通过 SOCKS 请求 GitHub HTTPS 验证端到端；关闭本任务代理后确认监听释放。预期不要求密码、不重复绑定、不公开代理端口。
- 本地预检查：AGENTS.md 已全面使用 Guqq2；目录不是 Git 仓库；现有任务记录目录不存在，本次建立。

- 2026-10-22 11:54:29 +08:00：实际结果：ssh -G 主机、用户、密钥、ProxyJump、回环 RemoteForward 符合要求；BatchMode SSH 成功；专用代理 curl 退出码 0，GitHub HTTP/2 200。文档命令与实际使用参数一致。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: conditional_trajectory implementation checks PASS: five target-MAE/missingness/paired/fixed-cohort/geometry tests, five native observer selection/RNG regression tests, compileall and bash -n on both Slurm wrappers. Planned GPU controls: source/checkpoint hashes, all native CFG embedding flags, same-forward score conversion, exact paired priors, full state coverage, prespecified terminal ALIGNN architecture controls, raw-state/physics-ledger audit. No GPU outcome claimed yet.

- 2026-10-30T16:29:45: Stage-analysis unit PASS: seven conditional_trajectory tests including two synthetic event/species-change/nonmonotonicity tests; compileall and phase.sbatch bash -n. Primary GPU505 is running; no outcomes inferred yet.

- 2026-10-30T16:40:45: Presentation/force-summary revision PASS: eight unit tests, compileall, bash -n and diff check. Original GPU505+CPU506/507 exit0;4872rawstate audit,356ALIGNN+356MatterSim points, six native ALIGNN controls exactzero error, native score conversion exactzero, downloaded archive SHA9c925ebf07a682198c6d65791c91ce1cb12a41b8e25bfa82ee61df943c4e1169 verified and artifact/raw hashes verified locally. Source revision changes reporting only; next full CPU rerender and visual QA.

- 2026-10-30T16:47:30: Manifest ownership fix PASS: ten tests, including later-stage overwrite isolation and missing-owned-output rejection; compileall/bash -n/diff checks pass. Full raw audit508/phase509 exit0 and all raw/prediction bytes unchanged. Prior artifact hash failure traced to renderer ownership and fixed before this commit; next CPU integration verifies both final manifests.

- 2026-10-30T16:52:08: FINAL PASS: CPU510 exit0; final archive SHAf9121d84643aa44577782d1e915835ae47289cd88b0b16f31895a2e3ebeb1eb1; both renderer manifests/source hashes verified;28raw/model/CSVfiles unchanged against original archive. PNG visual QA complete (2520x2160 and2340x1620), no empty panel or cropped labels. RESULTS table values/paths/Markdown checked; Unicode-minus formatting normalized for numeric comparison. Ten module tests and five native observer tests passed; generated evidence remains ignored.

- 2026-10-30T17:03:27: Availability documentation PASS:12HF model metadata records with file inventories; GSS config SHA matches publisher-pinned d408099ae279afcf40c913ee498672849e862d96fa15cb85dc1d3420c9adbe87; no GSS species corruption/loss and no property embeddings; OneScience gap config jointly corrupts species; electride CSP property embeddings empty; XPANCEO property keys e_above_hull/formula or dn/k_1064, not gap; official gap config inspected directly. README links and diff formatting checked. Opaque weights remain unverified, not declared incapable. No model tests needed for docs-only change.

- 2026-10-30T17:13:10: Training proposal documentation PASS: verified pinned finetune copies CSP diffusion config and can freeze shared weights; CSP weights cell1/pos0.1 with no species corruption/loss; composition-loader AST has no properties argument; scaler callback uses train loader; control mixing zero initialization verified. Proposed hyperparameters labeled unverified. Markdown table, relative README link, pinned official-source links and diff formatting checked. No training/GPU timing/accuracy claim or code execution.

- 2026-10-30T17:49:38: Adapter-design documentation correction PASS: AST confirms fixed NoiseLevelEncoding uses registered frequency buffer plus sine/cosine and no learnable parameter/linear layer; source confirms concatenated1024->512->512MLP, bias-free512mix, zero initialization and unconditional masking/zero encoding. Derived trainable control parameters4198400for4blocks checked. Diff formatting and plan references checked; no training or inference.

- 2026-10-30T19:14:57: Template-editing protocol checks PASS: pinned sampler prior reset/max_t/grid/dt and discrete-N binding verified; native last50/100/200grid starts0.05/0.10/0.20 with dt-0.001; source uses current types in D3PM posterior and native geometric marginals. Matched branches/budgets, endpoint mean, fixed species, equivalent-cell/locality and original-template cumulative distance, f(R(x)) selection and failures documented. README relative link/table/format checked. No editing inference/training performed.

- 2026-10-31T15:57:25: Reliability core/feature/head implementation local PASS:8tests for threshold boundaries/clamp/invalids, composition-invariant roles and original-train rejection, train-only scaling, selective bad-class metrics, empty/single-class cases, positive calibration slope, pickle-free identity/shape/roundtrip and MLP inference. compileall, both Slurm bash -n and diff checks passed. Native CUDAfeature/input-weight/reconstruction controls remain runtime integration gates. Local pandas optional dependency warnings nonfatal; no environment changes.

- 2026-10-31T16:03:46: Reliability report unit PASS:9tests including independent rank-statistic AUROC with ties; compileall/bash-n/diff check pass. Slurm511data PASS with69239source records and meta_train3540/meta_val760/calibration700/test4239,8190reducedcompositiongroups. Slurm512GPU active;4328/9239features at143.24s with exactlyzero same-forward reconstruction error. Final native/state/split/head audit pending.

- 2026-10-31T16:17:52: Diagnostic addition local PASS:10tests including selected-ID discrepancy logic, compileall/bash-n/diff checks. Initial pipeline511/512/513all0; archive0cb6aee601237955a313596af78b6117b5165a8f0c36c0193237879566f9d77f verified; full test AUROCs featureMLP0.8273/0.7790/0.7951; native512controls and reconstruction0. Posthoc GPUdiagnostic next on7published-output deltas, classifiers untouched.
