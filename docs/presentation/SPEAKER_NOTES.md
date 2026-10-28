# WaveSense presentation speaker notes

## Slide 1
Opening, 25 seconds: CFRP means carbon-fibre reinforced polymer. Our question is whether ultrasonic recordings distinguish a healthy plate from a controlled reference condition. We built a software prototype using public laboratory measurements. Do not claim we manufactured or tested the specimen ourselves.

## Slide 2
Explain the sensor-coordinate chart. Six transducers sit near each opposite edge of the 500 × 500 × 2 mm plate. Pairs send and receive waves. The two labels are healthy versus an attached reference disc. A surface disc is not a confirmed internal crack. Source: Moll et al. 2019, Tables 1–3, DOI 10.1038/s41597-019-0208-1.

## Slide 3
40 seconds: Supervised means the training examples have known answers. Binary means there are two classes. Despite its name, logistic regression is a classification algorithm. It is fast, suitable as a small-data baseline and interpretable. We use balanced class weights and regularization. The score is uncalibrated, so 0.9 must not be called a 90% certainty of damage. We do not use deep learning or unsupervised anomaly detection.

## Slide 4
40 seconds: RMS means root mean square and summarizes signal strength. The actual model uses four fixed windows: 150–350, 350–600, 600–950 and 950–1250 microseconds. 66 paths times 4 windows equals 264 features. Training-only scaling makes inputs comparable. This chart is a display-only 100-bin RMS summary of healthy/20181213T104208/40kHz, first path 1–2; it is not the four-window feature vector or raw waveform.

## Slide 5
45 seconds: The original model failed on hotter healthy recordings. Those previously seen recordings became development data, not a fresh test. We compared nine predefined configurations within grouped cross-validation. Ordinary logistic regression C=0.1 won the predeclared tie rule. We then froze it before downloading 40 additional acquisitions. Broader temperature coverage changed the training problem. Same-cycle recordings remain correlated; this is not independent-cycle or unseen-temperature validation. Evidence: docs/DEMO_PROTOCOL.md and outputs/presentation/frozen_model.json.

## Slide 6
40 seconds: Report counts rather than implying universal perfect accuracy. Eight healthy and 32 reference-object acquisitions were correct. Original model on exactly this cohort was 37/40, with three healthy false alarms. The original 25 hotter holdout failure is a separate historical experiment. Improvement cannot be attributed to a new temperature-correction algorithm because training coverage changed. Evidence: docs/DEMO_RESULTS.md and outputs/presentation/model.json.

## Slide 7
Live demo, 60 seconds: Open localhost:8501. Analyse example 61, the healthy case, then example 69, the D04 reference-object case. Explain the experiment label separately from the prediction. Rotate the plate and switch to Top. The orange position comes from documented experiment geometry, not ML localization. If the app fails, open outputs/presentation/healthy_report.html and reference_report.html. Finish: we built a reproducible classification workflow and an accessible interface; further validation is still required.
