# WARDEN

WARDEN uses byte frequencies to encode data under configurable DNA sequence
constraints, with path-check bases for inner decoding and column-wise
Reed-Solomon (RS) coding for recovery across strands.

This release contains WARDEN source code, the two paper inputs, shared
frequency tables, corpus audit evidence and reference DNA encodings.
WARDEN experiment configurations and reference values are provided in
`configs/paper_cases.json`. Experimental results and comparisons with
Explorer, DNA-Aeon and Gungnir are reported in the paper and its supplementary
tables; third-party baseline implementations are not included in this release.

## Installation

Use a separate Python 3.10+ environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/encode_us_gf256_second_rs_fixed130.py --help
python experiments/test_us_mod256_ids.py --help
python experiments/dataset_shared_table.py --help
```

See [RUNNING.md](RUNNING.md) for a complete encoding, channel-simulation
and decoding example. Settings and reference values for the 20 constraint
encodings, 36 path-check ablation trials and 45 recorded RS recovery trials
are listed in [configs/paper_cases.json](configs/paper_cases.json).
This JSON file is a case list, not an executable script. Use each case's
settings to adapt the example commands and choose a separate output directory.
Commands use Bash on Linux or WSL. Full runs can take many CPU-hours;
a GPU is not required.

Some entries in `configs/paper_cases.json` retain paths under `results/`
that identify original archived result files. Those files are not bundled
in this release, and those paths are not required inputs to the example
commands.

## Repository contents

| Path               | Contents                                                     |
| ------------------ | ------------------------------------------------------------ |
| `codec/`           | Core constrained entropy coding and decoding modules         |
| `experiments/`     | Encoding, noisy-channel evaluation/RS recovery and shared-table generation |
| `configs/`         | Input hashes, constraint definitions and the paper experiment list |
| `data/`            | Two source images, raw pixel-byte streams and shared-table audit evidence |
| `shared_tables/`   | The two frequency models used in the experiments             |
| `reference/`       | Archived WARDEN DNA encodings for checking new runs          |
| `RUNNING.md`       | Complete example commands and guidance for other experiment settings |
| `requirements.txt` | Runtime dependencies                                         |

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
tables in `shared_tables/`. 
