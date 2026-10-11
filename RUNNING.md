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

## Parameters

The experiment list is configs/paper_cases.json. The parameters for each experimen
t type are specified below and should be configured according to the corresponding
mappings.
