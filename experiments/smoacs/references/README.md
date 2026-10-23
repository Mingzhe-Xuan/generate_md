# Public MP reference acquisition

Download source: official materialsproject-build S3 Open Data, release2026-04-13.
This date falls within the paper's stated access interval but does not prove an
identical historical API snapshot. No API key is required. All derived selection,
decompression and structure processing runs in Slurm; source bytes are cached in
ignored results/smoacs/mp-public-reference. No API credentials are printed.

prepare.py reads selected Parquet columns through cached byte-range requests,
pins each object's ETag/size, and records hashes for every cached range. It collects
parent material IDs from the three paper thermo collections, then summary
structures/hull values. Parent material IDs need not equal every nested entry's
data.material_id; this remains an explicit provenance difference until audited.
Summary selection follows <=0.3eV/atom, and missing-hull elemental <=10sites.
All candidate chemical subsystems and all selected phases are retained without
random truncation. Unfiltered structures are also kept for novelty comparisons.

Install requirements.txt in a new Python3.10 venv .venv-smoacs-reference.
prepare.sbatch tests the proxy path inside its allocation; login-node loopback
access is never assumed. If unavailable, the job fails before network reads.
Reference structures subsequently require the same MatterSim5M relaxation and
MP2020Compatibility protocol as proposals before a hull can be evaluated.
Raw MP hull values select references; they are not the final MatterSim hull.

Source: https://docs.materialsproject.org/downloading-data/aws-opendata
