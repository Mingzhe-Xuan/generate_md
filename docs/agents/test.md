# 检查记录

## Guqq2 网络同步

- 2026-10-22 11:52:26 +08:00 计划：检查 ssh -G 的主机、用户、跳板机、密钥和 RemoteForward；BatchMode 验证公钥登录；ss 验证仅回环监听；通过 SOCKS 请求 GitHub HTTPS 验证端到端；关闭本任务代理后确认监听释放。预期不要求密码、不重复绑定、不公开代理端口。
- 本地预检查：AGENTS.md 已全面使用 Guqq2；目录不是 Git 仓库；现有任务记录目录不存在，本次建立。

- 2026-10-22 11:54:29 +08:00：实际结果：ssh -G 主机、用户、密钥、ProxyJump、回环 RemoteForward 符合要求；BatchMode SSH 成功；专用代理 curl 退出码 0，GitHub HTTP/2 200。文档命令与实际使用参数一致。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: conditional_trajectory implementation checks PASS: five target-MAE/missingness/paired/fixed-cohort/geometry tests, five native observer selection/RNG regression tests, compileall and bash -n on both Slurm wrappers. Planned GPU controls: source/checkpoint hashes, all native CFG embedding flags, same-forward score conversion, exact paired priors, full state coverage, prespecified terminal ALIGNN architecture controls, raw-state/physics-ledger audit. No GPU outcome claimed yet.

- 2026-10-30T16:29:45: Stage-analysis unit PASS: seven conditional_trajectory tests including two synthetic event/species-change/nonmonotonicity tests; compileall and phase.sbatch bash -n. Primary GPU505 is running; no outcomes inferred yet.
