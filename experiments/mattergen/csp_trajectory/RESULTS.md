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
