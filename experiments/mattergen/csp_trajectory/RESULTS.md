# GSS pure-CSP trajectory observations, 2026-10-30

Completed Slurm jobs: preparation492, native sampling/forces493, energy/force
controls495, independent CPU audit/report494; all exit0. Source sampling1be09c3,
force controlsf0b4622, analysis a57e896. All 12 GSS systems, two paths each,
native random geometry and fixed integer composition, 1000 PC steps. No reference
geometry, property conditioning, force guidance or relaxation enters sampling.

24,024 input states captured (prior + every predictor input); 21,985 MatterSim
single points compared, 2,039 geometry-guard exclusions, zero failed force calls.
6000 batched native forwards, fixed ordered species and RNG-neutral observation
checks passed. Main GPU elapsed1529.8s, including215.7s in native sampling.
Eight energy/force central-difference checks at four prespecified endpoints and
two step sizes passed, costing20 additional single points. Independent cosine
reconstruction maximum error4.51e-16; local NumPy decline/cohort audit passed.

| Metric | Cartesian score / force | Coordinate denoising direction / force |
|---|---:|---:|
| Mean at predictor900, same24paths | 0.397724 | 0.490640 |
| Mean at predictor999, same24paths | 0.681691 | 0.775027 |
| Downward adjacent transitions / valid transitions | 10131/20352 | 10070/20352 |
| Paths with a single-step decline >0.05 | 24/24 | 24/24 |
| Paths with a complete25-step-block mean decline >0.05 | 24/24 | 24/24 |
| Median maximum drawdown within contiguous valid segments | 1.229660 | 1.203086 |

Late average alignment increases, with substantial fluctuations in every path.
The average over eligible states in steps0–249 is negative (-0.097199 score,
-0.124090 coordinate direction). At steps900–999 the corresponding means are
0.500429 and0.622247, with complete24-path coverage. These are descriptive
observations from three native coupled batches, not independent-path confidence
intervals. Missing states are unknown and break adjacent-step comparisons.
Training overlap is unknown; this mechanism experiment does not estimate held-out
CSP recovery performance. Alignment is measured for coordinate vectors, not
lattice/stress updates.

Evidence: `results/mattergen/csp-trajectory-20261030` on Guqq2; verified local copy
`results/mattergen/evidence/csp-trajectory-20261030` includes REPORT.md,
report.json, dense NPZ states, single-point ledgers and PNG/SVG figures.
Archive SHA256: `c6b5bd56db6ca83c0437f5469da7072b6a474ed1ff34514dc51f3fc8f5b72167`.

## Energy trajectory follow-up

The same original dense trajectories contain21,963 predictor-input energies,
2,037 unknown predictor states and22 separate prior energies, for21,985 original
MatterSim single points. No new inference or relaxation was needed. The energy
plotter source is committed as0b0f4ed, independently matched to its runtime hash.
Output: `results/mattergen/evidence/csp-energy-trajectory-20261030`.

All24paths have a lower median E/atom at900–999 than at0–99. All24correlations
between step and25-step block-median energy are negative (Spearman range
-0.9697 to-0.5279). However,12/24have higher median energy at200–299 than at0–99:
the early shape is not shared by every path. Of20,352 genuinely adjacent valid
pairs,10,001increase (49.14%); in the last100steps,1098/2376increase (46.21%).
Thus the common pattern is an overall late energy decrease with substantial
stochastic fluctuations, rather than monotone energy minimization at each step.

One15-panel figure shows the available-path relative median/IQR, a fixed22-path
late cohort, coverage, and absolute E/atom for all12systems with two paths each.
Faint traces retain all evaluated states;25-step medians require20/25values and
missing blocks remain gaps. Symmetric-log axes retain negative energies and
large early spikes. The relative zero is the last observed predictor input999,
not a relaxed minimum or the native returned mean1000. Composition-specific
absolute energies are not ranked across systems. This original dense cohort is
not mixed with the numerically different later800/1000paired replay. Early noisy
state energies are ML-potential predictions and do not establish barrier heights.

All11unit tests, raw NPZ/geometry/energy/ledger/hash checks and independent
CSV/window/change-count audit pass; canonical PNG equals the visually inspected
preview byte-for-byte. Artifacts: `energy-trajectories.png/svg/pdf`, `energies.csv`,
`summary.json`, `REPORT.md` and `local-audit.json`.
PNG SHA256: `732b4129e9c65d3a98b44c287443ed79e674bce1e63f36e3db06e74a782ed654`.
CSV SHA256: `00edbea4cfc6354797cc6e15093f3b665519903673cbce63780a81b141e5ebe4`.

### Gaps and apparent midpoint discontinuity

Of960path/block combinations,107lack20/25valid points and have no solid median
segment;104ofthese are beforestep600. Raw missing predictor energies number2037.
The original symmetric-log energy axis visually emphasizes crossings nearzero;
zero absolute predicted energy is not a phase boundary. Block medians can also
switch rapidly when a block's high/low-energy population balance changes.

There is nevertheless a genuine broad decrease:21/24path medians decrease from
steps450–499 to550–599, with median paired change-5.2661eV/atom. The median
across path-window nearest-pair medians increases from0.9124 to1.1358 A, while
volume/atom medians remain about14 A^3. This is consistent with relief of close
contacts as noise decreases; it does not isolate coordinate/lattice/noise
contributions or establish a phase transition.

All seven reviewed native sampler/noise/corrector source hashes and the CSP
configuration hash match the actual run. There is no fixedstep500switch in
these sampler paths. For an8-atom cell, the smooth forward fractional positional
standard deviation is0.1125004/0.1118034/0.1111107 atsteps499/500/501; these are
not adaptive corrector step sizes. The largest adjacent50-step median-window
drop centers range70–655; they are window statistics, not exact transition times.

Diagnostic figure/JSON: `results/mattergen/evidence/csp-energy-diagnostic-20261030-final`.
Four panels show linear energy, closest distance, forward noise and display-gap
frequency. All14unit tests and raw/source/window arithmetic checks pass.
