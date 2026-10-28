# WaveSense: presentation and viva guide

## What machine learning do we use?

**Supervised binary classification using regularized logistic regression.** Supervised means we train using examples with known labels. Binary means two output classes: healthy-like and reference-damage-like. Logistic regression is a classification algorithm despite the word “regression” in its name. We use the scikit-learn implementation, not a neural network or deep learning model.

Each HDF5 recording has 66 sensor-pair responses. We remove the mean of each response and calculate its RMS signal strength in four fixed time windows. RMS means root mean square; it summarizes the size of a changing signal. We take log10 of those values, giving **66 × 4 = 264 features per recording**. Training-only scaling puts features on comparable scales. Logistic regression learns one weight per feature plus an intercept, combines the inputs, and converts the sum into a score between zero and one. Our fixed boundary is 0.5. The score is not calibrated and must not be presented as certainty of damage.

We use balanced class weights because there are more reference-object examples than healthy examples. Regularization discourages unnecessarily large weights. The parameter C controls its strength: smaller C means stronger regularization. Three-fold grouped cross-validation compares nine predefined choices: no temperature correction, linear healthy-temperature correction, or quadratic correction, each with C=0.01, 0.1, or 1. All fitting occurs inside training folds. The chosen method is frozen before new test recordings are downloaded.

The selected method for this run is **ordinary logistic regression, C=0.1, without extra temperature correction**. Multiple candidates tied in development cross-validation, so the predetermined rule chose the simpler model. The practical change is training across the available temperature range instead of only on cool recordings. Do not claim that a new temperature-compensation algorithm or TASP caused the improvement.

## A 90-second explanation

“Our project is WaveSense, a local application for exploring ultrasonic measurements from a carbon-fibre composite plate. Researchers collected these measurements using twelve transducers, which send and receive sound through the plate. We use a public dataset rather than claiming to have performed those laboratory experiments ourselves.

“Our machine-learning method is supervised binary classification using logistic regression. Each recording is converted into 264 signal-strength features. The model learns from known healthy and reference-object examples and predicts which class a new recording resembles.

“Our original model failed on hotter healthy recordings. We investigated temperature coverage, compared predefined configurations using grouped cross-validation, and froze the selected model before evaluating additional acquisitions. Our results sheet gives the actual counts and errors on that new test.

“The app lets us choose a recording, analyse it, inspect the signals, rotate a documented 3D plate and save a report. The orange point is a known experimental reference position, not a location found by our model. This is a laboratory classification prototype, not a certified detector for cracks or internal delamination.”

## Live demo: about two minutes

1. Double-click `Launch_WaveSense.cmd`. Keep its terminal window open. Use http://127.0.0.1:8501.
2. Select a test recording from the verified demo list in `docs/DEMO_RESULTS.md` and press **Analyse recording**. Start with the healthy case. Read the known condition and prediction separately.
3. Show a reference-object test case. Explain what the attached disc represents. If a listed example is wrong, show the error honestly.
4. Rotate the plate, switch to **Top**, and select another sensor connection. Explain that this changes the displayed measurement, not the classifier's inputs.
5. Expand **See the recorded sound**. Explain that temperature can also change the waveform.
6. Save the HTML report. If the live app fails, open the prepared HTML reports under `outputs/presentation/` in a browser; these contain actual saved predictions and plots.

The current app needs no internet once recordings and models are prepared. Have the report and slides downloaded locally before class.

## Likely questions

**Is this anomaly detection?** In an informal sense we flag a response resembling the reference condition. Technically, this implementation is supervised two-class classification, not unsupervised anomaly detection or a one-class novelty detector.

**Why logistic regression?** It is a simple baseline suitable for a small dataset, fast on a CPU and interpretable: its weights let us show which path features influence the result. We do not claim it is the best possible algorithm.

**What is the damage?** A removable aluminium disc attached to the plate surface by the original researchers. We cannot equate that with a real internal crack or delamination.

**Does the 3D image show predicted damage?** No. It shows documented geometry, sensors and the known experimental reference position. We have not implemented localization, crack size or depth estimation.

**How did you avoid leakage?** We group all measurements from each acquisition, fit preprocessing inside training folds, freeze model choices before opening the new test signals, and verify no acquisition groups or file hashes overlap. Neighboring samples from the same cycles can still be correlated; the test is not an independent-cycle validation.

**What changed from the failed model?** The original trained at <=38 °C and was tested at >=42 °C. The revised development set spans the available temperatures. Its new acquisition test asks a different, easier question. We compare both models on the same additional cohort, but cannot attribute improvement solely to an algorithmic change.

**Does 100% on a small test mean it is perfect?** No. Always state the numerator and denominator, same-specimen/cycle scope and the lack of real-delamination validation. A small test has substantial uncertainty, especially for the healthy class.

**What is our contribution?** Data ingestion and provenance, validation, feature extraction, model training/evaluation, signal comparisons, documented 3D visualization and a simple local workflow. The raw experiments were performed by the dataset authors. TASP is exploratory, not a demonstrated novel contribution in this demo.

**What comes next?** More healthy examples and independent specimens/cycles, realistic internal defects, better temperature robustness and score calibration. A future localization method would need its own implementation and validation.

## Sources

- Dataset and geometry: https://doi.org/10.1038/s41597-019-0208-1
- Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Leakage prevention: https://scikit-learn.org/stable/common_pitfalls.html
- Local evidence: `docs/DEMO_PROTOCOL.md`, `docs/DEMO_RESULTS.md`, `outputs/presentation/model.json`.
