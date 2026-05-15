# Fairness Audit Notes

Required checks before deployment:

- Acceptance rate by `Sex` and age band.
- Demographic parity difference and ratio.
- True positive and false positive rates by protected group.
- Equalized odds difference.
- Written justification for any fairness/performance tradeoff.

Recommended mitigation options:

- Threshold adjustment by policy review group where legally allowed.
- Fairlearn Exponentiated Gradient or Grid Search in a future dependency-approved phase.
- Feature review for protected proxies.
