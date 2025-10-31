# Machine Learning Based Classification of Aggressive and Malignant Renal Tumors from Multimodal Data

This repository contains the implementation used in the paper  
**"Machine Learning Based Classification of Aggressive and Malignant Renal Tumors from Multimodal Data"**,  
available on [MedRxiv](https://www.medrxiv.org/content/10.1101/2025.02.04.25321687v1.full.pdf).

The project integrates radiological and clinical data to classify renal tumors into three categories: **Benign**, **Indolent Malignant**, and **Aggressive Malignant** using contrastive learning representations.  
It supports both **image-based** and **embedding-based (MLP)** pipelines.


All parameters are set in `configs/default.yaml`. Make sure to set data.mode to "embeddings" or "images" depending on which pipeline you want to run.

How to Run the Code
Activate your environment:
conda activate env
Run the nested cross-validation:
```bash
for fold_test in 0 1 2 3 4
do
  echo ">>> OUTER TEST FOLD = $fold_test"

  for fold_inner in 0 1 2 3 4
  do
    echo "--- Training INNER FOLD $fold_inner ---"
    python src/main.py \
      --config configs/default.yaml \
      --fold_test $fold_test \
      --fold_inner $fold_inner
  done

  echo "--- Training FINAL MODEL for OUTER FOLD $fold_test ---"
  python src/main.py \
    --config configs/default.yaml \
    --fold_test $fold_test
done
---
The outer loop defines the five test sets.
The inner loop performs cross-validation for hyperparameter tuning.
After the inner folds are done, the model is retrained on the combined train+validation data for that outer fold.
When all folds are done, run the post-processing:
```bash
python src/extract_logits_main.py
python src/postprocess_all.py
---
This will collect all predictions, compute metrics (ROC-AUC, PR-AUC, Brier score, etc.), and generate the final plots and result tables.

---
