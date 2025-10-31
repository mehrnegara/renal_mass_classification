#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=train_nested_embeddings.log

module purge
eval "$(conda shell.bash hook)"
conda activate gpuenv-1

echo "✅ Running Nested CV (Embeddings Mode Only)"
echo " Must have: data.mode: embeddings in configs/default.yaml"
START_TIME=$(date +%s)
# loop over ALL OUTER test folds
for fold_test in 0 1 2 3 4
do
  echo "===================================="
  echo ">>> OUTER TEST FOLD = $fold_test"
  
  # 1) TRAIN ALL INNER FOLDS (validation models only)
  for fold_inner in 0 1 2 3 4
  do
    echo "--- Training INNER FOLD $fold_inner (for thresholding only) ---"
    python src/main.py \
      --config configs/default.yaml \
      --fold_test $fold_test \
      --fold_inner $fold_inner
  done

  # 2) TRAIN FINAL OUTER MODEL (train+val full fold, no inner fold)
  echo "--- Training FINAL MODEL for OUTER FOLD $fold_test ---"
  python src/main.py \
    --config configs/default.yaml \
    --fold_test $fold_test

done

echo "✅ All nested folds trained successfully."
python src/extract_logits_main.py
python src/postprocess_all.py
END_TIME=$(date +%s)
echo "✅ Total runtime: $((END_TIME - START_TIME)) seconds"
