# Running WARDEN

Install the dependencies as described in [README.md](README.md). This guide
provides one complete paper-trial example and the parameter changes needed
for Figures 2 and 3. Run commands from the repository root in Bash on Linux
or WSL. The example uses the full Alzheimer input, not a small smoke test;
it can take substantial CPU time. A GPU is not required.

## One complete example

This example uses Alzheimer, B1 constraints, one path-check base, RS N=255
and P=44 (K=211), nominal 1% IDS, no dropout and seed 201. It encodes the
input once, simulates the noisy channel, decodes, and reports exact recovery.

```bash
(
set -eu
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
OUT=outputs/example_alzheimer_s201
if [ -e "$OUT" ]; then
  printf '%s\n' "Output already exists: $OUT; choose a new OUT directory."
  exit 1
fi
mkdir -p "$OUT"

python experiments/encode_us_gf256_second_rs_fixed130.py \
  --data data/alzheimer.bin \
  --code-dir codec \
  --height 128 \
  --width 128 \
  --transform row_first \
  --N 255 \
  --K 211 \
  --dna-length 130 \
  --pb 3 \
  --gtable-log 14 \
  --shared-table shared_tables/alzheimer.json \
  --signature-count 1 \
  --homopolymer-h 4 \
  --gc-window 10 \
  --gc-min 4 \
  --gc-max 6 \
  --forbidden-motifs AGCT,GACGC,CAGCAG,GATATC,GGTACC,CTGCAG,GAGCTC,GTCGAC,AGTACT,ACTAGT,GCATGC,AGGCCT,TCTAGA \
  --candidate-symbols 128 \
  --out-dir "$OUT/encoded"

python experiments/test_us_mod256_ids.py \
  --data data/alzheimer.bin \
  --code-dir codec \
  --height 128 \
  --width 128 \
  --transform row_first \
  --N 255 \
  --K 211 \
  --dna-length 130 \
  --pb 3 \
  --gtable-log 14 \
  --shared-table shared_tables/alzheimer.json \
  --signature-count 1 \
  --homopolymer-h 4 \
  --gc-window 10 \
  --gc-min 4 \
  --gc-max 6 \
  --forbidden-motifs AGCT,GACGC,CAGCAG,GATATC,GGTACC,CTGCAG,GAGCTC,GTCGAC,AGTACT,ACTAGT,GCATGC,AGGCCT,TCTAGA \
  --oligos "$OUT/encoded/oligos.csv" \
  --all \
  --workers 1 \
  --beam 256 \
  --cost-window 4 \
  --short-lookahead 4 \
  --lookahead-pool 1.5 \
  --pi 0.0033333333333333335 \
  --pd 0.0033333333333333335 \
  --ps 0.0033333333333333335 \
  --dropout 0.0 \
  --seed 201 \
  --cost-threshold 16 \
  --rs-feedback-iterations 32 \
  --out "$OUT/result.json"

python -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); print("Exact recovery:", r["rs_recovery"]["end_to_end_image_exact"])' "$OUT/result.json"
)
```

Outputs are `outputs/example_alzheimer_s201/encoded/oligos.csv`, the adjacent
`summary.json`, and `outputs/example_alzheimer_s201/result.json`. 

## Parameters for Figures 2 and 3

The experiment list is `configs/paper_cases.json`. It contains **20 constraint
encodings, 36 ablation trials and 45 recorded RS-recovery trials**, with their
parameters and reference records. It is a manifest, not an executable script;
the original programs do not automatically read it. Adapt the example using
the mapping below and run the listed cases individually.

| Manifest/configuration item | Command-line setting |
| --- | --- |
| `dataset` | Set `--data`, `--height`, `--width`, `--transform` and `--shared-table` from the input table below |
| `P` | Keep `--N 255`; set `--K` to 255-P in both commands |
| `signature_count` | Set `--signature-count` in both commands |
| `config` | Expand the selected constraint setting into flags in both commands; see the constraint table below |
| `ids_percent` | Set `--pi`, `--pd`, `--ps` to `ids_percent / 300`; use the acceptance threshold below |
| `dropout_percent` | Set `--dropout` to `dropout_percent / 100` |
| `seed` | Set decoder `--seed` to the listed value: 201, 202 or 203 |
| `encode_only` | When true, run only the encoder |
| Output paths | Use a new directory/file per case; point `--oligos` to its matching encoding |

### Inputs and common settings

| Input | `--data` | Height x width | `--transform` | `--shared-table` |
| --- | --- | --- | --- | --- |
| Alzheimer | `data/alzheimer.bin` | 128 x 128 | `row_first` | `shared_tables/alzheimer.json` |
| Chest | `data/chest.bin` | 299 x 299 | `column_first` | `shared_tables/chest.json` |

The input bytes are raw pixels. Differences are computed modulo 256 and
restarted independently per row or column. RS data are zero-padded, arranged
row-major, coded per column in GF(256), then flattened row-major.

Keep the example's 130-nt coding-region length, `--pb 3`, table log 14,
payload probe limit 128, beam 256, edit penalty 4, cost window 4, lookahead
4/1.5, maximum 32 RS feedback iterations and one decoder worker. The encoder's
128-byte payload probe limit is not an alphabet limit; the decoder considers
all byte values subject to index and stream bounds.

| Nominal IDS | `--pi`, `--pd`, `--ps` (each) | `--cost-threshold` |
| --- | --- | --- |
| 0.5% | `0.0016666666666666668` | 8 |
| 1% | `0.0033333333333333335` | 16 |

Dropout settings are `0.0` and `0.01`. An encoding can be reused when only
the channel probabilities or seed change. Re-encode when changing the input,
model, constraints, P or check count, and give the decoder matching settings.

### Figure 2: constraints

Encode both inputs under A1--D3, with P=55 (K=200) and one path-check base.
This gives 20 deterministic encodings; no noisy decoding is required.

| Configuration | `--homopolymer-h` | `--gc-window` | `--gc-min` | `--gc-max` | Forbidden motifs |
| --- | --- | --- | --- | --- | --- |
| A1 | 8 | 0 | Omit | Omit | None |
| A2 | 4 | 6 | 2 | 4 | None |
| A3 | 4 | 6 | 2 | 4 | M13 |
| B1 | 4 | 10 | 4 | 6 | M13 |
| B2 | 4 | 10 | 5 | 5 | M13 |
| C1 | 3 | 6 | 2 | 4 | M13 |
| C2 | 4 | 6 | 2 | 4 | ML10 |
| D1 | 4 | 6 | 2 | 4 | M13 + ML10 |
| D2 | 4 | 8 | 3 | 5 | M13 + ML10 |
| D3 | 4 | 10 | 4 | 6 | M13 + ML10 |

The allowed maximum homopolymer length is h-1. GC bounds apply to complete
sliding windows. Replace `--forbidden-motifs` with the comma-separated
sequences in `configs/constraints.json`, not the names M13/ML10; for the union,
include all 23 motifs. ML10 includes `TTTATA`. For A1/A2 remove the entire
`--forbidden-motifs` argument. A1 also omits GC bounds and sets window 0.
M13 scans for A1/A2 are diagnostic only, not violations of enabled constraints.

### Figure 3: ablation and recovery

| Experiment | Constraints/check bases | RS setting | Channel | Runs |
| --- | --- | --- | --- | --- |
| Path-check ablation | B1; check counts 0, 1, 2 | Alzheimer P=44; Chest P=52 | Nominal 1% IDS; dropout 0% or 1% | Seeds 201/202/203 for each combination; 36 trials |
| RS recovery | B1; one check base | P values listed in `configs/paper_cases.json` | Nominal 0.5% or 1% IDS; dropout 0% or 1% | 45 recorded trials; use each listed seed |

The ablation supports the single-read accuracy and code-rate panels. Changing
the check count also changes payload capacity and strand count; it is not a
fixed total-base-budget comparison. Some failing RS settings stopped before
all seeds; do not invent missing trials. The eight reported passing points
each achieved exact recovery for all three seeds:

| Nominal IDS (%) | Dropout (%) | Alzheimer P | Chest P |
| --- | --- | --- | --- |
| 0.5 | 0 | 20 | 24 |
| 0.5 | 1 | 24 | 32 |
| 1 | 0 | 44 | 44 |
| 1 | 1 | 44 | 52 |

These are the smallest passing P among the tested settings, not proven global
minima. Selection and reporting used the same three seeds.

## Checking results

| Quantity | New output / calculation | Reference under `results/warden/` |
| --- | --- | --- |
| DNA and payload intervals | `oligos.csv`: `dna`, `start_symbol`, `end_symbol_exclusive`, `symbol_count` | DNA files named in `constraints.csv`; archived encodings under `reference/` |
| Code rate | 8 x input bytes / total coding-region bases | `constraints.csv`, `ablation_summary.csv`, `recovery_points.csv` |
| Constraint compliance | Scan complete GC windows, homopolymers and active motifs against the selected setting | `constraints.csv` |
| Correct/wrong/failed read fractions | `exact_rate`, `wrong_rate`, `erasure_rate` in `result.json` | `ablation_trials.csv` |
| Exact recovery after inverse preprocessing | `rs_recovery.end_to_end_image_exact` | `ablation_trials.csv`, `recovery_grid_index.csv` |

Code rates include indices, check bases, RS parity and termination, but exclude
shared-table storage and primers. Use sample standard deviation (n-1) for
three-run summaries. Per-trial aggregate records are under
`results/warden/records/`. Compare DNA, byte intervals, read counts, event
totals and RS metrics; timings and absolute paths are not reproducibility targets.

Use the same code, data, tables, parameters, seeds and recorded dependency
versions. Seed equality alone does not guarantee cross-platform identity.
Input/model hashes are in `configs/datasets.json`; package hashes are in
`SHA256SUMS.json`. Documentation replacement requires updating its document
entries without changing hashes for unchanged experimental files.

## Channel definition

Nominal IDS is `pi + pd + ps`. Per original nucleotide, substitution is sampled
only if the nucleotide is not deleted; insertion is sampled separately. Thus
the expected event count for length L is `L * (pd + (1-pd)*ps + pi)`.
WARDEN perturbs strands with strand-specific random streams, shuffles them,
then samples independent dropout using another stream. Single-read fractions
exclude dropped strands. The evaluator derives its global inner-symbol bound
from the encoded pool and uses the known source for scoring; it is not a
metadata-free decoder for externally sequenced reads.

## Shared-table checks

Figures 2 and 3 use the supplied `shared_tables/*.json`; rebuilding is not
required. `data/table_evidence/` contains the audited image lists, hashes,
histograms and numerical model-match reports. The original builder is
`experiments/dataset_shared_table.py`; its `--help` lists the corpus input and
output options. 
