# Verified presentation results

Model: supervised logistic regression, C=0.1; 264 log-RMS features; fixed threshold 0.5.

Development: 60 recordings. Fresh acquisition test: 40 recordings. The model was frozen before test download.

**40/40 correct: 8/8 healthy, 32/32 reference-object cases; zero healthy false alarms.**

The original model scored 37/40 on this same cohort, with 3/8 healthy false alarms. Training temperature coverage changed; this is not evidence for a new temperature-correction algorithm.

**Scope:** same specimen and recording cycles, with different timestamps. Not independent-cycle or unseen-temperature validation. The reference object is a surface disc, not real internal delamination.

## Demo examples and offline reports

- Example 61: healthy, 20 °C, `healthy/20181213T104208/40kHz` → **Healthy-like**. Offline report: `outputs/presentation/healthy_report.html`.
- Example 69: D04, 20 °C, `D04/20190104T123220/40kHz` → **Reference-damage-like**. Offline report: `outputs/presentation/reference_report.html`.

## All test predictions

| Recording | Condition | °C | Prediction |
|---|---|---:|---|
| healthy/20181213T104208/40kHz | healthy | 20.5 | Healthy-like |
| healthy/20181214T180429/40kHz | healthy | 43.0 | Healthy-like |
| healthy/20181215T224025/40kHz | healthy | 53.9 | Healthy-like |
| healthy/20181217T022327/40kHz | healthy | 31.5 | Healthy-like |
| healthy/20181218T105231/40kHz | healthy | 31.1 | Healthy-like |
| healthy/20181219T164921/40kHz | healthy | 53.6 | Healthy-like |
| healthy/20181220T205607/40kHz | healthy | 43.5 | Healthy-like |
| healthy/20181222T020741/40kHz | healthy | 20.6 | Healthy-like |
| D04/20190104T123220/40kHz | D04 | 20.5 | Reference-damage-like |
| D04/20190105T035131/40kHz | D04 | 31.5 | Reference-damage-like |
| D04/20190105T201207/40kHz | D04 | 43.0 | Reference-damage-like |
| D04/20190106T094356/40kHz | D04 | 54.0 | Reference-damage-like |
| D04/20190106T224306/40kHz | D04 | 54.4 | Reference-damage-like |
| D04/20190107T110500/40kHz | D04 | 43.5 | Reference-damage-like |
| D04/20190108T012154/40kHz | D04 | 32.0 | Reference-damage-like |
| D04/20190108T162700/40kHz | D04 | 20.5 | Reference-damage-like |
| D12/20181223T151347/40kHz | D12 | 20.6 | Reference-damage-like |
| D12/20181224T065209/40kHz | D12 | 31.5 | Reference-damage-like |
| D12/20181224T224517/40kHz | D12 | 43.0 | Reference-damage-like |
| D12/20181225T115522/40kHz | D12 | 54.0 | Reference-damage-like |
| D12/20181226T004717/40kHz | D12 | 54.5 | Reference-damage-like |
| D12/20181226T133514/40kHz | D12 | 43.5 | Reference-damage-like |
| D12/20181227T033846/40kHz | D12 | 32.1 | Reference-damage-like |
| D12/20181227T190054/40kHz | D12 | 20.5 | Reference-damage-like |
| D16/20181230T124806/40kHz | D16 | 20.5 | Reference-damage-like |
| D16/20181231T043850/40kHz | D16 | 31.6 | Reference-damage-like |
| D16/20181231T201254/40kHz | D16 | 43.0 | Reference-damage-like |
| D16/20190101T093435/40kHz | D16 | 54.0 | Reference-damage-like |
| D16/20190101T224222/40kHz | D16 | 54.5 | Reference-damage-like |
| D16/20190102T110924/40kHz | D16 | 43.5 | Reference-damage-like |
| D16/20190103T011253/40kHz | D16 | 32.0 | Reference-damage-like |
| D16/20190103T153025/40kHz | D16 | 20.6 | Reference-damage-like |
| D24/20190108T205853/40kHz | D24 | 20.6 | Reference-damage-like |
| D24/20190109T125228/40kHz | D24 | 31.5 | Reference-damage-like |
| D24/20190110T053226/40kHz | D24 | 43.1 | Reference-damage-like |
| D24/20190110T185001/40kHz | D24 | 54.0 | Reference-damage-like |
| D24/20190111T080707/40kHz | D24 | 54.5 | Reference-damage-like |
| D24/20190111T211034/40kHz | D24 | 43.5 | Reference-damage-like |
| D24/20190112T111504/40kHz | D24 | 32.0 | Reference-damage-like |
| D24/20190113T014336/40kHz | D24 | 20.5 | Reference-damage-like |

Model version: `temperature-b4ef927fbdf7`.
Frozen model SHA-256: `f5f6f01ceda01bde5f619c4ff4b951a42673e958c51454a1b65c44945f8f86f5`.
