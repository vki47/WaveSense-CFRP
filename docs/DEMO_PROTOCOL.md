# Presentation model protocol — 28 October 2026

The original 60 acquisitions have already been inspected, including the old high-temperature test results. They are now explicitly **development data** for a second experiment. The original model and its failed high-temperature test remain preserved in outputs/model. They are not an untouched test for this upgrade.

Before opening any new test signals, freeze this procedure:

- Input: the same native 40 kHz / 10 MHz recordings, 66 paths, four predeclared log-RMS windows.
- Candidates: logistic regression with no temperature correction, linear healthy-temperature correction, or quadratic healthy-temperature correction. For each, C is 0.01, 0.1, or 1. Class weights are balanced; classification threshold is fixed at 0.5.
- Use three-fold stratified group cross-validation (seed 47) within the original 60 acquisitions. Keep all paths of an acquisition together. Fit healthy temperature trends, scalers and classifier only inside each training fold.
- Select by pooled out-of-fold balanced accuracy; ties by fewer healthy false alarms, then simpler temperature trend and stronger regularization. Cross-validation numbers are development/selection results, not final accuracy claims.
- Refit the selected configuration on the 60 development acquisitions. Freeze its coefficients and model checksum before reading new test signals.
- Additional test cohort: eight previously unused acquisition timestamps from each published condition archive, equally spaced through the eligible sorted timestamps after excluding the development timestamps. No signal, temperature or outcome is used to choose those timestamps. Download using bounded ranges and record SHA-256/ZIP CRC provenance. Do not substitute another cohort based on results.
- Evaluate the frozen model and original baseline on the same new 40-acquisition cohort once. Keep all errors and report class counts, recall, false alarms, balanced accuracy and macro F1. This is a same-specimen, same-cycle acquisition holdout, **not** an independent-cycle or unseen-temperature validation. Nearby acquisitions remain correlated. Do not claim real delamination detection or general structural safety.
- The second experiment deliberately changes from low-temperature extrapolation to training across the available temperature range. Any comparison must disclose that both temperature coverage and processing may differ; improvement cannot be attributed solely to the correction algorithm.

If downloading or evaluation fails, retain the original demo and report the actual blocker. No fabricated scores and no threshold adjustment based on the new test.
