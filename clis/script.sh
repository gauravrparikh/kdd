#!/bin/bash
#SBATCH --job-name=parcel_partition
#SBATCH --output=logs/parcel_partition_%A_%a.out
#SBATCH --error=logs/parcel_partition_%A_%a.err
#SBATCH --array=0-399    # adjust based on total combos0-1999
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL


source .venv/bin/activate 

# -------------------------------
# Define parameter grid
# -------------------------------
MODELS=("linear_regression" "decision_tree" "random_forest" "gradient_boosting" "svr")
MIN_LEAFS=(10 25 50 75 100 150 200 250 350 500 750 1000 1500 2000 2500 3500 5000 7500 10000 15000)
SEEDS=(1)
OUTCOMES=("LANDVALUE" "synthetic_1_rho_0.9" "synthetic_2_rho_0.5" "synthetic_3_rho_0.1" )\
          #  "synthetic_4_rho_0.6" "synthetic_5_rho_0.5" "synthetic_6_rho_0.4" "synthetic_7_rho_0.3" \
          #  "synthetic_8_rho_0.2" "synthetic_9_rho_0.1" "BLDGVALUE" "VALUATION" "CALC_ACRES" \
          #  "FinArea" "YearBlt" "TAXSTAMPS")

# Compute total combinations
NUM_MODELS=${#MODELS[@]}
NUM_LEAFS=${#MIN_LEAFS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_OUTCOMES=${#OUTCOMES[@]}
TOTAL=$((NUM_MODELS * NUM_LEAFS * NUM_SEEDS * NUM_OUTCOMES))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds total $TOTAL combinations."
  exit 1
fi

# Compute indices for this job
idx=$SLURM_ARRAY_TASK_ID
model_idx=$(( idx % NUM_MODELS ))
idx=$(( idx / NUM_MODELS ))
leaf_idx=$(( idx % NUM_LEAFS ))
idx=$(( idx / NUM_LEAFS ))
seed_idx=$(( idx % NUM_SEEDS ))
idx=$(( idx / NUM_SEEDS ))
outcome_idx=$(( idx % NUM_OUTCOMES ))

MODEL=${MODELS[$model_idx]}
MIN_LEAF=${MIN_LEAFS[$leaf_idx]}
SEED=${SEEDS[$seed_idx]}
OUTCOME=${OUTCOMES[$outcome_idx]}

echo "[INFO] Job $SLURM_ARRAY_TASK_ID running:"
echo "Model=$MODEL, Leaf=$MIN_LEAF, Seed=$SEED, Outcome=$OUTCOME"

uv run partition_expt.py \
  --parcels-folder /usr/xtmp/gr90/Spatial/real_world_data/parcels \
  --streets-folder /usr/xtmp/gr90/Spatial/real_world_data/streets \
  --target-col "$OUTCOME" \
  --model "$MODEL" \
  --min-samples-leaf "$MIN_LEAF" \
  --seed "$SEED"

