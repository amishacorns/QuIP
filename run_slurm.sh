#!/bin/bash
#SBATCH --job-name=p_sweep_g128_incoh
#SBATCH -o /home/jad443/slurm_logs/%x_%A_%a.out
#SBATCH -e /home/jad443/slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jad443@cornell.edu
#SBATCH --mem=32000
#SBATCH -n 2
#SBATCH -t 3:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH --array=0-10%1

export HF_TOKEN=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

MODEL="facebook/opt-125m"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Generate p_values from 0 to 0.50 with 0.05 increments
p_values=($(seq 0 0.05 0.50))

# Function to compute lookup values
compute_lookup_values() {
    local p=$1
    local temp_values=(-1 -p p 1)
    local values=()
    for val in "${temp_values[@]}"; do
        new_val=$(echo "$val * 1.5 + 1.5" | bc)
        values+=("$new_val")
    done
    echo "${values[@]}"
}

# Compute lookup values for each p value
lookup_values_array=()
for p in "${p_values[@]}"; do
    lookup_values_array+=("$(compute_lookup_values $p)")
done

# Get the current lookup values based on the SLURM array task ID
lookup_values=${lookup_values_array[$SLURM_ARRAY_TASK_ID]}

echo "Running model with lookup_values=($lookup_values)"

/home/jad443/.conda/envs/compressor/bin/python opt.py \
    ${MODEL} wikitext2 --group_size=8 \
    --wbits=2 --incoh_processing --lookup_values "$lookup_values" \
    --timestamp "${TIMESTAMP}_p${p_values[$SLURM_ARRAY_TASK_ID]}"
