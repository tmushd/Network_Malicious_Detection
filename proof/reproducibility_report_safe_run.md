# Reproducibility Report: run_20260412_112604

## Model Metrics

| task       | model   |   accuracy |   precision |   recall |   f1_weighted |   f1_macro |   eval_rows | device   |
|:-----------|:--------|-----------:|------------:|---------:|--------------:|-----------:|------------:|:---------|
| binary     | ELECTRA |   0.99     |    0.990035 | 0.99     |      0.989977 |   0.988744 |         500 | cpu      |
| binary     | LGBM    |   0.956667 |    0.956551 | 0.956667 |      0.956501 |   0.951078 |        1500 | nan      |
| binary     | RF      |   0.960667 |    0.960886 | 0.960667 |      0.960366 |   0.95527  |        1500 | nan      |
| binary     | XGB     |   0.96     |    0.960113 | 0.96     |      0.959728 |   0.954584 |        1500 | nan      |
| multiclass | ELECTRA |   0.988    |    0.987985 | 0.988    |      0.987916 |   0.985882 |         500 | cpu      |
| multiclass | LGBM    |   0.925333 |    0.925146 | 0.925333 |      0.923749 |   0.857813 |        1500 | nan      |
| multiclass | RF      |   0.936667 |    0.936433 | 0.936667 |      0.934274 |   0.878553 |        1500 | nan      |
| multiclass | XGB     |   0.928    |    0.927758 | 0.928    |      0.926179 |   0.859992 |        1500 | nan      |

## Paper vs Reproduced

| task       | model   | metric      |   paper |   reproduced |    abs_diff |
|:-----------|:--------|:------------|--------:|-------------:|------------:|
| multiclass | LGBM    | accuracy    |    0.96 |     0.925333 | 0.0346667   |
| multiclass | LGBM    | precision   |    0.95 |     0.925146 | 0.0248544   |
| multiclass | LGBM    | recall      |    0.93 |     0.925333 | 0.00466667  |
| multiclass | LGBM    | f1_weighted |    0.94 |     0.923749 | 0.0162506   |
| multiclass | XGB     | accuracy    |    0.96 |     0.928    | 0.032       |
| multiclass | XGB     | precision   |    0.96 |     0.927758 | 0.0322419   |
| multiclass | XGB     | recall      |    0.94 |     0.928    | 0.012       |
| multiclass | XGB     | f1_weighted |    0.94 |     0.926179 | 0.0138213   |
| multiclass | RF      | accuracy    |    0.97 |     0.936667 | 0.0333333   |
| multiclass | RF      | precision   |    0.96 |     0.936433 | 0.0235673   |
| multiclass | RF      | recall      |    0.95 |     0.936667 | 0.0133333   |
| multiclass | RF      | f1_weighted |    0.95 |     0.934274 | 0.0157255   |
| multiclass | ELECTRA | accuracy    |    0.99 |     0.988    | 0.002       |
| multiclass | ELECTRA | precision   |    0.99 |     0.987985 | 0.00201531  |
| multiclass | ELECTRA | recall      |    0.99 |     0.988    | 0.002       |
| multiclass | ELECTRA | f1_weighted |    0.99 |     0.987916 | 0.00208389  |
| binary     | LGBM    | accuracy    |    0.93 |     0.956667 | 0.0266667   |
| binary     | LGBM    | precision   |    0.94 |     0.956551 | 0.0165511   |
| binary     | LGBM    | recall      |    0.92 |     0.956667 | 0.0366667   |
| binary     | LGBM    | f1_weighted |    0.93 |     0.956501 | 0.0265013   |
| binary     | XGB     | accuracy    |    0.94 |     0.96     | 0.02        |
| binary     | XGB     | precision   |    0.95 |     0.960113 | 0.0101128   |
| binary     | XGB     | recall      |    0.92 |     0.96     | 0.04        |
| binary     | XGB     | f1_weighted |    0.94 |     0.959728 | 0.0197281   |
| binary     | RF      | accuracy    |    0.95 |     0.960667 | 0.0106667   |
| binary     | RF      | precision   |    0.96 |     0.960886 | 0.000886059 |
| binary     | RF      | recall      |    0.95 |     0.960667 | 0.0106667   |
| binary     | RF      | f1_weighted |    0.95 |     0.960366 | 0.0103663   |
| binary     | ELECTRA | accuracy    |    0.99 |     0.99     | 0           |
| binary     | ELECTRA | precision   |    0.99 |     0.990035 | 3.52782e-05 |
| binary     | ELECTRA | recall      |    0.99 |     0.99     | 0           |
| binary     | ELECTRA | f1_weighted |    0.99 |     0.989977 | 2.25563e-05 |
