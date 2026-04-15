# Week 12 Novelty Report

## What changed

- Replaced the plain standalone baseline block with a new hybrid fusion block.
- The new block combines Random Forest lexical probabilities with pretrained ELECTRA probabilities.
- A logistic-regression meta-model is trained on the validation split and evaluated on held-out test data.

## Binary Metrics

| task   | model         |   accuracy |   precision |   recall |   f1_weighted |   f1_macro |
|:-------|:--------------|-----------:|------------:|---------:|--------------:|-----------:|
| binary | ELECTRA       |   0.993333 |    0.9934   | 0.993333 |      0.993317 |   0.9925   |
| binary | HYBRID_FUSION |   0.993333 |    0.9934   | 0.993333 |      0.993317 |   0.9925   |
| binary | RF            |   0.95     |    0.950753 | 0.95     |      0.949387 |   0.942712 |

## Comparison Against Parent Paper ELECTRA Accuracy

| model         | paper_reference_model   |   paper_accuracy |   reproduced_accuracy |   improvement_vs_paper |
|:--------------|:------------------------|-----------------:|----------------------:|-----------------------:|
| RF            | ELECTRA                 |             0.99 |              0.95     |            -0.04       |
| ELECTRA       | ELECTRA                 |             0.99 |              0.993333 |             0.00333333 |
| HYBRID_FUSION | ELECTRA                 |             0.99 |              0.993333 |             0.00333333 |

## Key takeaway

- Hybrid fusion binary accuracy: 0.9933
- Parent paper ELECTRA binary accuracy: 0.9900
- Improvement vs parent paper binary accuracy: 0.0033
