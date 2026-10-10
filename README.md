# WARDEN

WARDEN uses byte frequencies to encode data under configurable DNA sequence
constraints, with path-check bases for inner decoding and column-wise
Reed-Solomon (RS) coding for recovery across strands.

This repository contains WARDEN source code, the two paper inputs, shared
frequency tables and final experimental results. Explorer, DNA-Aeon and
Gungnir are provided as results and configuration records only; their
implementations are not included.

## Installation

Use a separate Python 3.10+ environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/encode_us_gf256_second_rs_fixed130.py --help
python experiments/test_us_mod256_ids.py --help
python experiments/dataset_shared_table.py --help
```

`test_us_mod256_ids.py` is the actual noisy-channel decoding and evaluation
program, despite its historical filename. It is required for recovery experiments.

See [RUNNING.md](RUNNING.md) for one complete encoding/decoding example, the
Figure 2 and Figure 3 parameter tables, and result comparisons. The full case
list remains in `configs/paper_cases.json`; it is not an executable script.
Commands use Bash on Linux or WSL. Full runs can take many CPU-hours;
a GPU is not required.

## Repository contents

| Path | Contents |
| --- | --- |
| `codec/` | Core constrained entropy coding and decoding modules |
| `experiments/` | Encoding, noisy-channel evaluation/RS recovery and shared-table generation |
| `configs/` | Input hashes, constraint definitions and the paper experiment list |
| `data/` | Two source images, raw pixel-byte streams and shared-table audit evidence |
| `shared_tables/` | The two frequency models used in the experiments |
| `reference/` | Archived WARDEN DNA encodings for checking new runs |
| `results/warden/` | Final WARDEN summaries and per-trial aggregate records |
| `results/comparison/` | Final comparison tables, trial records and protocol parameters |
| `requirements.txt` | Runtime dependencies |
| `SHA256SUMS.json` | Packaged-file checksums |
| `LICENSE_PENDING.md` | Pending code-license decision; not an open-source license |

## Input data and shared tables

The `.bin` files contain raw 8-bit grayscale pixels, not compressed files or
image-container bytes. Their equality to the decoded source images was checked.
Preprocessing is applied by WARDEN, not pre-applied to these files.

| Input | Recorded original image | Shape | Source bytes | WARDEN preprocessing |
| --- | --- | --- | --- | --- |
| Alzheimer | `mild_595.jpg` | 128 x 128 | 16,384 | Row-wise first differences modulo 256 |
| Chest | `COVID-1734.png` | 299 x 299 | 89,401 | Column-wise first differences modulo 256 |

The author identifies the image sources as **Alzheimer MRI Preprocessed
Dataset** and **COVID-19 Radiography Database**. The supplied file-level audit
identifies the following local corpora used to check the shared tables:

- Alzheimer: 4,435 images under `alzheimer/train/train`, in Mild_Demented,
  Non_Demented and Very_Mild_Demented directories.
- Chest: 9,161 images under `nashidul/Chest_Radiography_Database`, in COVID-19,
  Normal, Tuberculosis and Viral Pneumonia categories.

These are the audited local collections, not a claim that every image in each
named public dataset was used. The exact upstream download/version attribution
was not recovered; the Chest directory name alone does not identify a specific
release of the cited database.

`data/table_evidence/` contains image manifests and hashes, aggregate histograms,
audit reports and environment/archive checksums. The 2026-10-01 audit decoded
all listed images without errors and reproduced the saved models' numerical
symbols, frequencies, cumulative starts and table log. Both test images are
included in these corpora; the frequency models are not held-out models.
The audit records the corpus at audit time, not a recovered historical build
log. Matching normalized tables does not uniquely prove historical membership;
the original build command and table-log search were not reconstructed.

The full corpora are not bundled. The two-input experiments use the saved
tables in `shared_tables/`; see [RUNNING.md](RUNNING.md#shared-table-checks)
for the optional builder and audit evidence. Historical absolute paths in table
metadata are provenance only; the experiment commands use repository-relative
paths.

## Comparison results

Use `results/comparison/comparison_results.csv` for the final summary,
`formal_trials.csv` for per-seed results, `sanity_checks.csv` for no-noise checks,
and `protocols/` within that directory for full baseline parameters.

| Method | Evaluated configuration | Dropout conditions |
| --- | --- | --- |
| WARDEN | B1; 130 nt; one path-check base; N=255; P=44/52 for Alzheimer/Chest | 0% and 1% |
| Explorer | One continuous DNA sequence per input; published repair path | 0% only; 1% is NE |
| DNA-Aeon | M13-excluding codebook; 18-byte chunks; extra-packet ratio 0.057; four threads | 0% and 1% |
| Gungnir | Released decoder; 130 nt; density 0.8; thread1=thread2=1 | 0% and 1% |

Formal evaluated conditions use nominal 1% total IDS and seeds 201, 202, 203.
Only the final M13-excluding DNA-Aeon configuration is included. Its codebook
was deterministically filtered from the official codebook, not generated by
running the authors' Java tool; construction and decoder details are recorded
in `results/comparison/protocols/dnaaeon_m13.json`. Native strand lengths differ
between methods. Explorer dropout is **NE (not evaluated)**, not 0/3.

The comparison tables can be inspected and replotted, but this repository does
not independently rerun the baselines. Gungnir's final results use completed
continuation records; timeouts are not counted as evaluated decoding failures.
These results describe the tested configurations, not each method's optimum.

## Reproducibility and release notes

- Local checks used Python 3.12.14, NumPy 2.3.5, reedsolo 1.7.0 and Pillow
  12.3.0 on Windows. `requirements.txt` gives compatibility ranges, not the
  original server's complete locked environment.
- Core-source and data hashes, command parsing, numerical shared-table
  reconstruction and archived-result consistency were checked. A small
  file-level test recovered the input and matched deterministic metrics in
  two same-seed noisy runs. Full medical-image experiments were not rerun
  for documentation/package simplification; cross-platform identity is not proven.
- No automatic resume, plot generation or result-comparison service is
  included. [RUNNING.md](RUNNING.md) explains the output fields and comparisons.
- Code rates exclude shared-table storage and primers. An extract's
  `_archive.original_sha256` identifies its full original record;
  `SHA256SUMS.json` checks the bundled files themselves. Replacing documentation
  requires updating that manifest; older documentation hashes will no longer match.
- The code license remains pending. Authors must also check third-party data
  redistribution terms before public release. If images cannot be redistributed,
  provide exact acquisition/preprocessing instructions and hashes instead;
  do not claim immediate reproduction after silently removing required inputs.
