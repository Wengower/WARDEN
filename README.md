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

The image sources are **Alzheimer MRI Preprocessed
Dataset** and **COVID-19 Radiography Database**. The supplied file-level audit
identifies the following local corpora used to check the shared tables:

- Alzheimer: 4,435 images under `alzheimer/train/train`, in Mild_Demented,
  Non_Demented and Very_Mild_Demented directories.
- Chest: 9,161 images under `nashidul/Chest_Radiography_Database`, in COVID-19,
  Normal, Tuberculosis and Viral Pneumonia categories.

`data/table_evidence/` contains image manifests and hashes, aggregate histograms,
audit reports and environment/archive checksums. 

The full corpora are not bundled. The two-input experiments use the saved
tables in `shared_tables/`; see [RUNNING.md](RUNNING.md#shared-table-checks)
for the optional builder and audit evidence. 
