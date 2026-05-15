# Model Card: German Credit Risk Platform

## Intended Use

Support credit risk analytics teams in evaluating default risk for German Credit loan applicants.

## Primary Metrics

- Bad-class recall
- Bad-class F2-score
- AUC-ROC
- Business-cost optimized threshold

## Fairness Review

Protected attributes are `Age` and `Sex`. Production approval requires review of demographic parity and equalized odds metrics.

## Limitations

The dataset is small, historical, and may encode biased lending decisions. Do not use this model for automated adverse action without legal, compliance, and risk approval.
