
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

- 2026-10-31T22:19:28: Fix observed ID-less trainingCSV handling, preserve516failure; new metadata makes unavailable IDalignment explicit. No reconstructed targetcount uses that CSV. All16tests PASS; rerun exact60000training-label audit in fresh retry02 directory.

- 2026-10-31T22:23:35: Training-gap distribution measured and reported: exactmode0eV35.15%; allmedian0.665eV, positivemedian1.88285eV; 2.5+/-0.2eV3194(5.32%). Positiveexactmode~0.0023eV only20repeats, not a useful representative target. Document tentative GRPO implications and zero-clipped reward ties without launching RL. Evidence local results/mattergen/evidence/alignn-training-gap-mode-20261031.

- 2026-10-31T22:37:21: Add grpo module README/interfaces and audited targetcohortprepare (realDFT2.4-2.6, original60000trainIDs, <=20atoms, groupedroles/rejections/provenance). DirectpretrainedonlineGRPO clarified. All3cohorttestsPASS. Initialsafe nextstep CPUcohort audit and base/checkpoint/cacheinventory; no RL results claimed.

- 2026-10-31T22:58:35: Add explicitaugmented-actionPCpolicy, rank4LoRA, GPUbasemodel/conditional loaders, onlinegroups/frozenALIGNNreward and full1000levelreplay+realrewardGRPOsmoke. Changes fromnativePC documented (expectednormcorrector,percrystalsteps,cellprojection, stdfloors); nativeconditionalbaseline retained for finalcomparison. No propertyimprovement or trainedpolicy claimedbeforeGPUgate. CPUrecovery can avoid461MBVlabupload onlyifall281base tensorsmatch; source-tested11contractsPASS.

- 2026-10-31T23:25:40: Add train.py/train.sbatch (directbaseGRPO,noSFT), 6400rollouts, 2innerupdates, rank4LoRAAdamW2e-6cosinedecay, referenceKL, invalidrewardpreservation, validationonlybestselection, sourcecheckedresume. Add losslessofficialdelta reconstruction with all126chunk/281tensorproof; preserve519failure. Vlabfailedfullupload,6waychunks andserialintermittentlinks; completedchunksreused, directGit/HF/CDNboundeddiagnosticsfailed. Newserialupload makesverifiedtransportprogress28/126 andiscurrentlylive; actualtraining/comparison remainspending.

- 2026-10-31T23:46:50: Implement compare.py/compare.sbatch for actualrequested trainedpolicy-vs-conditional test, including matchedkernel and officialnative controls, finalcheckpoint transparency, canonicalrelaxationrawforcecheck, proxyadherence limitation, invalidfull-denominators, compositiondiversity and samplingbatchbootstrap.20unit testsPASS. Fullobjective remainsunfulfilleduntil exactweightsGPUsmoke/training/finalcomparison succeed; activeassettransfer progresses95of126chunks.

- 2026-10-31T23:50:17: Add nativePCbase andnativePCselectedGRPO controls to prevent sampler choice fromdetermining apparentcheckpoint superiority. Sevenarm plan256each includes raw/commonMatterSimpropertytests, compositiondiversity andbatch-awareuncertainty. NativeGRPOcheckpoint uses fixedtraining-kernelvalidationselection, never finaltestselection.21localtestsPASS; networkassettransfer is onlycurrentlyliveprocess.

- 2026-11-01T00:05:18: Resourcebottleneck resolved with exactofficialdelta; sourcebase nowbitwiseverified, notconditionalbranch. Fixmodels.py verifiedsource importpath omittedwhile reusing verify_source helper. Preserve522failure; nextsubmitnewGPUonegroupfull1000levelsmoke only, thenauthorizedtraining afterpass.

- 2026-11-01T00:18:28: Resourceidentityall281andGPUmechanicsbothverified; policy nowhasactualonegroupGRPOevidence, notjustlocalcontracts. Alignsmokewithtwoinnerupdates andfixoraclecallcounter withoutchangingrewards/samplingmath. Plan autoafterok pipeline to keepfullobjective moving: strict2-updateGPUgate ->200x32onlinecandidates ->sevenarm256each independentcomparison/commonMatterSim. No benchmark superiority claimedfromsmoke.

- 2026-11-01T00:29:35: Officialbase/data/21unitcontracts/fullGPU2-updategate complete. Durablepipeline524->525->526submitted;525liveRTX5090trainingjob (initialvalidation16/64),526dependencyqueuedforsevenarms256eachpluscanonicalMatterSim. Formalbudget6400newrollouts and independentfinalevaluationremainpending; do not infermodelimprovementfromsmoke. Activeusergoalpreservedunmodified.

- 2026-11-01T00:41:26: Add checkpoint_io.py and CPUtestsbatch; comparison and futuretrainresume nowread NumPy64LR checkpts withoutdisablingweights_only or changingcorevalidatedsource. This prevents delayedcheckpointreadfailures afterlongtraining.525keptlive;526keptqueued. Local23testsPASS; actualserverroundtrip next.

- 2026-11-01T00:50:14.760496: Training525 has completed 3/200 rollout iterations (96 generated candidates) and is sampling iteration4; comparison526 remains afterok525. All first96 candidates were evaluable; gap passes per32 were1,1,0. Fixed initial validation was0/64 gap passes, MAE1.9062167414eV. Training batches are different samples and do not establish an improvement. First paired validation is scheduled at iteration25. Actual checkpoint acceptance528 and environment IO tests527 passed without changing the four GPU-validated runtime files. Continue existing jobs and inspect fixed-seed validation before judging efficacy.

- 2026-11-01T00:53:55.523076: Fixed queued comparison response gate so MatterSim CUDA OOM and other explicit infrastructure failures abort assessment rather than become material failures. Duplicate worker results also reject. Existing training525 and comparison526 dependency unchanged; validated training files untouched. Last authoritative status525: rollout5, 4 completed iterations/128 candidates; fixed validation25 remains pending.

- 2026-11-01T00:55:56.465832: Completed comparison integrity fix and server acceptance529 without GPU contention or restarting525. All25 local tests and7 server comparison tests passed. Training advanced to160 candidates/5 completed iterations; no final efficacy evidence yet. Continue original525/526 pipeline.
