# PathSnap experiment notes

This project compares five candidate classifiers, tunes Random Forest on trip-grouped training folds, and evaluates HMM/Viterbi decoding on the student-test partition.

## Workflow

Run from the repository root:

```powershell
python scripts/preprocess.py
python scripts/explore.py
python scripts/train.py
python scripts/tune.py
python scripts/evaluate.py
python scripts/visualize.py
```

The evaluator partition is reserved for the final demonstration and is not used for training, tuning, or development metrics. Student-test is a development set.

`model_comparison.csv` reports both candidate-row test accuracy and observation-level Top-1 accuracy for each classifier. `tuned_vs_untuned.csv` records the route-level tuning comparison. `route_decoder_comparison.csv` compares classifier and HMM Top-1 accuracy per saved model and trip.

After the model is locked using training-fold results, run `python scripts/evaluate_final.py` once to record aggregate inference accuracy on `evaluator_test`. Do not use that result to select or tune a model.
