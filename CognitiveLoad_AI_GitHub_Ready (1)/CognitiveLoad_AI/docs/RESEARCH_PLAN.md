# Research plan

## Working title

**Multimodal Behavioral Signals for Generalizable Cognitive Load Estimation**

## Research question

How does combining behavioral signals affect the reliability of cognitive-load estimation across different users?

## Research questions

**RQ1.** How well can individual behavioral signals estimate cognitive load?

**RQ2.** Does combining behavioral signals improve estimation?

**RQ3.** Does the improvement remain when evaluating participants not seen during training?

**RQ4.** Which modalities contribute most consistently across participants?

## Study design

Use controlled task conditions with multiple difficulty levels and repeated participants. Pair task difficulty with subjective workload and objective task performance rather than assuming a single proxy is ground truth.

## Modeling

Start with interpretable baselines such as logistic regression. Report multimodal fusion through an ablation matrix. Later experiments can add tree-based models or temporal models after establishing the baseline.

## Evaluation

The primary split should be participant-grouped. Candidate metrics are accuracy, macro-F1, ROC-AUC, calibration/error analysis, and confidence intervals. Report participant count and number of windows explicitly.

## Deliverables

1. Public demo application.
2. Reproducible feature-extraction pipeline.
3. Participant-level dataset and data dictionary.
4. Ablation/evaluation notebook or script.
5. Research paper with limitations and reproducibility notes.
