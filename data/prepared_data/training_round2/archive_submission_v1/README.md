# Validation splits of the initial submission (archived)

These `val.csv` files were used to train the BERT models of the initial PLOS Digital Health submission (git tag `submission-v1`).
Because of a bug in the split code (`val_idx, val_idx = next(...)` in `model/datahandler.py`, introduced in commit `932e0c9`, 4 Jul 2024), each of them is **identical to the corresponding `test.csv`**:

| Task | Rows |
|---|---|
| data_collection | 96 |
| data_type | 96 |
| number_of_participants | 96 |
| sex_of_participants | 96 |
| relevant | 565 |

All other tasks were not affected.

The corrected `val.csv` files in the task folders were rebuilt on 2 Nov 2026 by rerunning the original split (`prepare_splits`, seed 42) with the fixed code. The rebuilt `train.csv` and `test.csv` were byte-identical to the frozen files, so only `val.csv` changed: it is now the held-out half that had never been used (full data − train − test), disjoint from train and test.
For `relevant`, the corrected validation split has 564 rows (5,642 records = 4,513 train / 564 val / 565 test); the earlier `Val_size` of 565 in `meta.json` was the size of the test copy.

The test sets are unchanged; their checksums are in `../test_checksums.sha256`.
