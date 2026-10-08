# SHL Hiring Assessment 2026 – Spoken English Grammar Scoring

## Overview

This project was developed for the **SHL Hiring Assessment 2026**. The objective is to predict a continuous spoken-English grammar score between **0 and 5** from approximately **45–60 second audio recordings**.

The solution combines **speech transcription, linguistic features, acoustic features, and transcript-based regression** to estimate grammar scores.

## Approach

The final pipeline consists of four main stages:

### 1. Speech Transcription

Cached **Whisper transcripts** were used for the training and test audio recordings.

Missing transcripts were replaced with empty strings before feature extraction.

### 2. Feature Engineering

A total of **56 features** were used:

- **12 linguistic features** extracted from transcripts
- **44 acoustic features** extracted from audio

#### Linguistic features

The linguistic feature set includes:

- Character count
- Word count
- Unique word count
- Type-token ratio
- Average word length
- Maximum word length
- Sentence count
- Average sentence length
- Maximum sentence length
- Filler-word count
- Filler-word ratio
- Repeated-word ratio

#### Acoustic features

The acoustic feature set includes:

- Audio duration
- RMS energy statistics
- Zero-crossing rate statistics
- Spectral centroid statistics
- Spectral bandwidth statistics
- Spectral rolloff statistics
- MFCC 1–13 mean and standard deviation
- Silence ratio
- Speech transitions
- Transition rate
- Peak amplitude

## Models

Two complementary regression models were used.

### ExtraTreesRegressor

ExtraTrees was trained on the combined acoustic and linguistic feature matrix.

The model was configured with:

- 1500 trees for final training
- `max_features=0.8`
- `min_samples_leaf=2`
- `random_state=42`

ExtraTrees captures nonlinear relationships between the extracted audio and linguistic features.

### TF-IDF + Ridge

The transcript text was represented using both:

- Word-level TF-IDF features with unigram and bigram representations
- Character-level TF-IDF features using character n-grams

These representations were combined and passed to a Ridge regression model.

This model captures lexical and character-level patterns present in the spoken transcripts.

## Ensemble

The final prediction combines the two models:

```text
Final Prediction =
    0.68 × ExtraTrees Prediction
  + 0.32 × Ridge Prediction
```

The resulting predictions are clipped to the valid grammar-score range of **0 to 5**.

## Validation

A shuffled **5-fold cross-validation** procedure was used to generate out-of-fold predictions.

The final ensemble achieved:

| Metric | Result |
|---|---:|
| Pearson Correlation | **0.8365** |
| RMSE | **0.7126** |

The out-of-fold results provide a more realistic estimate of generalization performance than in-sample training metrics.

### Training Performance

The final model achieved a training RMSE of:

**0.0969**

This is an **in-sample training metric** and is therefore expected to be optimistic. The 5-fold out-of-fold results above are the more meaningful validation measurements.

## Final Submission

The final model was retrained using all available labeled training data and generated predictions for all **216 test samples**.

The submission file contains:

```text
filename
label
```

The final submission file was saved as:

```text
/kaggle/working/submission.csv
```

The best submitted Kaggle result achieved a public leaderboard score of:

**0.5567**

## Visualizations

The notebook includes:

1. **Top 20 ExtraTrees Feature Importances**  
   Shows the features with the highest contribution to the ExtraTrees model.

2. **Actual vs Predicted Training Scores**  
   Compares the final ensemble's training predictions against the actual grammar scores.

## Repository Contents

```text
SHL-Grammar-Scoring-2026/
│
├── shl-grammar-scoring-v2-improved.ipynb
└── README.md
```

The notebook contains the complete preprocessing, feature engineering, model training, validation, evaluation, visualization, and submission pipeline.

## Reproducibility

The notebook was developed in Kaggle using the **SHL Hiring Assessment 2026** competition data and the cached transcript dataset.

The competition audio and private dataset files are **not included in this public repository**.

## Key Results

- **56 engineered features**
- **ExtraTrees + TF-IDF/Ridge ensemble**
- **68/32 model weighting**
- **5-fold OOF Pearson:** 0.8365
- **5-fold OOF RMSE:** 0.7126
- **Training RMSE:** 0.0969
- **Best Kaggle public score:** 0.5567
- **Test samples:** 216
