# CSP benchmark dataset preparation

Downloads the GSS fixed-composition polymorph reference set and the standard
DiffCSP MP20 train/validation/test CSVs. Source revisions, exact byte lengths,
Zenodo MD5 and Git blob SHA1 values are pinned in `sources.json`. Generated
assets and evidence stay in ignored `results/mattergen/csp-benchmarks-20261029`.
No dataset, archive or generated inventory is committed through ordinary Git.

`download.py` is network I/O only (at most two curl downloads), so it may run on
the login node under project policy. Use the single healthy dedicated SSH proxy
and disable forwarding on ordinary working connections. Direct mode requires
the user's existing direct-download permission. Downloads resume into partial
files and retain original bytes. Complete-length files are verified in Slurm
before they are trusted.

```bash
.venv-mattergen/bin/python experiments/mattergen/csp_benchmarks/download.py \
  --root results/mattergen/csp-benchmarks-20261029 \
  --proxy socks5h://127.0.0.1:1080 --submit-verification
```

`--submit-verification` queues exactly one `verify.sbatch` after downloads.
Otherwise submit it explicitly:

```bash
sbatch --export=ALL,CSP_BENCHMARK_ROOT=results/mattergen/csp-benchmarks-20261029 \
  experiments/mattergen/csp_benchmarks/verify.sbatch
```

`verify.py` checks all source bytes/hashes, safely unpacks the GSS archive and
checks that every indexed reference has exactly one CIF. It inventories all CSV
rows/IDs, duplicate IDs and cross-split ID intersections without changing any
source row or filtering model-unsupported compositions. It parses one CIF per
GSS system and the first/last CIF of each MP20 split as format probes. This is
representative format verification, not full-dataset structure standardization.
The CIF column preserves full cell composition; `pretty_formula` is not used to
infer the atom count. The additional author `cif.conv` column is retained.

Output interfaces:

- `downloads/gss_reference_structures.tar.gz`: unchanged author release.
- `gss/gss_reference_structures/<System>/*.cif` and `index.csv`: original
  12-system reference collection (4919 structures), including RSS frequencies.
- `mp20/train.csv`, `val.csv`, `test.csv`: exact author splits.
- `file-manifest.json`, `gss-summary.json`, `mp20-summary.json`: integrity,
  counts, locations and split-ID overlaps.
- `structure-probes.json`, `structure-probes/`: reproducible parsing checks.
- `complete.json`, `preparation-evidence.tar.gz` and SHA256: Slurm source/job
  provenance and small metadata evidence for local independent checking.

The current CSP checkpoint was trained on Alex-MP20. These files do not by
themselves establish that MP20 test targets were absent from that training set.
Cross-split ID checks within MP20 are separate from checkpoint-exposure and
structural-equivalence audits. No generation, scoring, relaxation, SUN, DFT or
training is performed by this preparation module.

Reuses existing `.venv-mattergen`; no environment/package changes required.

```bash
python -m unittest discover -s experiments/mattergen/csp_benchmarks -p 'test_*.py'
```
