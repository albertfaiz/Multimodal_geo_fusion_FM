#!/bin/bash
#SBATCH --job-name=reviewer_suite
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=reviewer_suite_%j.log
#SBATCH --error=reviewer_suite_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fxa230012@utdallas.edu

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"

# THREAD CONTENTION SAFEGUARDS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ENVIRONMENT SETUP
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate geo_ml

cd $SLURM_SUBMIT_DIR

echo "--- Executing Diagnostic Pipeline ---"
python3 -u 01_reviewer_diagnostics_ml.py

echo "--- Executing Parallel Multi-Learner (R3.8) ---"
python3 -u 05_multilearner_parallel_hpc.py

echo "--- Executing NTL + PM2.5 Ablation ---"
python3 -u 03_nightlights_pm25_ablation.py

echo "--- Executing SHAP Interpretation ---"
python3 -u 02_reviewer_diagnostics_shap.py

echo "All reviewer tasks finished at $(date)"
