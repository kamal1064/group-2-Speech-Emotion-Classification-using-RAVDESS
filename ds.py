
"""
Speech Sentiment Recognition
3 classes: Negative, Neutral, Positive

The script:
1. Reads the RAVDESS dataset.
2. Converts the original 8 emotions into 3 sentiment classes.
3. Keeps actors separate for train/validation/test.
4. Extracts audio features.
5. Adds a few simple training augmentations.
6. Scales and selects useful features.
7. Compares a few RBF-SVM settings.
8. Evaluates the selected model on unseen actors.
"""

import os
import random
import warnings
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----------------------------
# Settings
# ----------------------------

SEED = 42
DATASET_PATH = Path(
    r"C:\Users\kamal\Downloads\Audio_Speech_Actors_01-24_16k"
)
OUTPUT_DIR = Path("outputs_best_sentiment")

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")
OUTPUT_DIR.mkdir(exist_ok=True)


# Actors are kept separate so the model is tested on people it has
# never seen during training.
TRAIN_ACTORS = [f"{i:02d}" for i in range(1, 17)]
VAL_ACTORS = [f"{i:02d}" for i in range(17, 21)]
TEST_ACTORS = [f"{i:02d}" for i in range(21, 25)]


EMOTION_NAMES = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

# RAVDESS has 8 emotions. For this project they are grouped into
# three broader sentiment categories.
SENTIMENT_MAP = {
    "Neutral": "Neutral",
    "Calm": "Positive",
    "Happy": "Positive",
    "Surprised": "Positive",
    "Sad": "Negative",
    "Angry": "Negative",
    "Fearful": "Negative",
    "Disgust": "Negative",
}

SENTIMENT_TO_INT = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

INT_TO_SENTIMENT = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]


# ----------------------------
# Dataset information
# ----------------------------

def get_emotion_code(filepath):
    """Read the emotion code from a RAVDESS filename."""
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("-")

    if len(parts) < 7:
        return None

    return parts[2]


def get_emotion(filepath):
    """Convert the RAVDESS emotion code to its name."""
    code = get_emotion_code(filepath)

    if code not in EMOTION_NAMES:
        return None

    return EMOTION_NAMES[code]


def get_sentiment(filepath):
    """Convert an emotion into Negative, Neutral, or Positive."""
    emotion = get_emotion(filepath)

    if emotion is None:
        return None

    return SENTIMENT_MAP[emotion]


def get_actor(filepath):
    """Read the actor number from a RAVDESS filename."""
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("-")

    if len(parts) < 7:
        return None

    return parts[-1]


def build_dataframe(dataset_path):
    """Find all valid WAV files and store their metadata."""
    records = []

    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(root, filename)
            emotion = get_emotion(filepath)
            sentiment = get_sentiment(filepath)
            actor = get_actor(filepath)

            if emotion and sentiment and actor:
                records.append(
                    {
                        "path": filepath,
                        "emotion": emotion,
                        "sentiment": sentiment,
                        "actor": actor,
                    }
                )

    return pd.DataFrame(records)


def show_distribution(name, dataframe):
    """Print the number of samples in each sentiment class."""
    print(f"\n{name}")
    print("-" * 40)

    counts = dataframe["sentiment"].value_counts()

    for sentiment in SENTIMENT_ORDER:
        print(f"{sentiment:<10}: {counts.get(sentiment, 0)}")


# ----------------------------
# Audio augmentation
# ----------------------------

def add_noise(y):
    """Add a small amount of random noise."""
    noise = np.random.normal(0, 0.003, len(y))
    return y + noise


def pitch_shift(y, sr):
    """Move the pitch slightly up or down."""
    steps = np.random.uniform(-1.5, 1.5)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)


def time_stretch(y):
    """Make the audio slightly faster or slower."""
    rate = np.random.uniform(0.90, 1.10)
    return librosa.effects.time_stretch(y=y, rate=rate)


# ----------------------------
# Feature extraction
# ----------------------------

def extract_features_from_signal(y, sr):
    """Turn one audio signal into a fixed-length feature vector."""
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 1000:
        return None

    features = []

    # MFCCs and their first/second derivatives
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for values in (mfcc, mfcc_delta, mfcc_delta2):
        features.extend(np.mean(values, axis=1))
        features.extend(np.std(values, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.std(zcr)])

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.std(rms)])

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    for values in (centroid, bandwidth, rolloff):
        features.extend([np.mean(values), np.std(values)])

    # Pitch / fundamental frequency
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        f0 = f0[~np.isnan(f0)]

        if len(f0):
            features.extend(
                [
                    np.mean(f0),
                    np.std(f0),
                    np.min(f0),
                    np.max(f0),
                    np.median(f0),
                ]
            )
        else:
            features.extend([0, 0, 0, 0, 0])

    except Exception:
        # Some recordings may not produce a usable pitch estimate.
        features.extend([0, 0, 0, 0, 0])

    return np.asarray(features, dtype=np.float32)


def load_audio(filepath):
    """Load an audio file as mono at 16 kHz."""
    return librosa.load(filepath, sr=16000, mono=True)


# ----------------------------
# Build feature matrices
# ----------------------------

def extract_normal_dataset(dataframe):
    """Extract features without augmentation."""
    X, y = [], []

    for i, (_, row) in enumerate(dataframe.iterrows(), start=1):
        try:
            signal, sr = load_audio(row["path"])
            features = extract_features_from_signal(signal, sr)

            if features is not None:
                X.append(features)
                y.append(SENTIMENT_TO_INT[row["sentiment"]])

        except Exception as error:
            print(f"Could not process {row['path']}: {error}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(dataframe)}")

    return np.asarray(X), np.asarray(y)


def extract_training_dataset(dataframe):
    """
    Extract original features plus three augmented versions:
    noise, pitch shift, and time stretch.
    """
    X, y = [], []

    for i, (_, row) in enumerate(dataframe.iterrows(), start=1):
        try:
            signal, sr = load_audio(row["path"])
            label = SENTIMENT_TO_INT[row["sentiment"]]

            versions = [
                signal,
                add_noise(signal),
                pitch_shift(signal, sr),
                time_stretch(signal),
            ]

            for audio in versions:
                features = extract_features_from_signal(audio, sr)

                if features is not None:
                    X.append(features)
                    y.append(label)

        except Exception as error:
            print(f"Training error for {row['path']}: {error}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(dataframe)}")

    return np.asarray(X), np.asarray(y)


# ----------------------------
# Main program
# ----------------------------

print("=" * 70)
print("SPEECH SENTIMENT RECOGNITION")
print("3-CLASS POSITIVE / NEGATIVE / NEUTRAL")
print("=" * 70)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATASET_PATH}")

print("\nDataset found successfully!")
print(DATASET_PATH)

df = build_dataframe(DATASET_PATH)
print(f"\nTotal valid audio files: {len(df)}")

# Keep actors completely separate.
train_df = df[df["actor"].isin(TRAIN_ACTORS)].copy()
val_df = df[df["actor"].isin(VAL_ACTORS)].copy()
test_df = df[df["actor"].isin(TEST_ACTORS)].copy()

print("\n" + "=" * 70)
print("ACTOR-INDEPENDENT SPLIT")
print("=" * 70)

print("Training actors   :", TRAIN_ACTORS)
print("Validation actors :", VAL_ACTORS)
print("Test actors       :", TEST_ACTORS)

print("\nTraining files   :", len(train_df))
print("Validation files :", len(val_df))
print("Testing files   :", len(test_df))

show_distribution("TRAINING DISTRIBUTION", train_df)
show_distribution("VALIDATION DISTRIBUTION", val_df)
show_distribution("TEST DISTRIBUTION", test_df)


print("\n" + "=" * 70)
print("EXTRACTING TRAINING DATA")
print("=" * 70)

X_train, y_train = extract_training_dataset(train_df)

print("\n" + "=" * 70)
print("EXTRACTING VALIDATION DATA")
print("=" * 70)

X_val, y_val = extract_normal_dataset(val_df)

print("\n" + "=" * 70)
print("EXTRACTING TEST DATA")
print("=" * 70)

X_test, y_test = extract_normal_dataset(test_df)

print("\nFeature extraction complete!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# Replace invalid numerical values before scaling.
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)


# Scale using training data only.
print("\nScaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Select the most useful features using mutual information.
print("\nSelecting best features...")

K_FEATURES = min(300, X_train_scaled.shape[1])

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=K_FEATURES,
)

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print("Original features :", X_train.shape[1])
print("Selected features :", X_train_selected.shape[1])


# ----------------------------
# Train SVM models
# ----------------------------

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

counts = Counter(y_train)

for class_id in range(3):
    print(INT_TO_SENTIMENT[class_id], ":", counts[class_id])


print("\n" + "=" * 70)
print("TRAINING RBF SVM MODELS")
print("=" * 70)

models = [
    SVC(
        kernel="rbf",
        C=3,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    ),
    SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    ),
    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    ),
    SVC(
        kernel="rbf",
        C=5,
        gamma=0.01,
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    ),
]

best_model = None
best_val_accuracy = -1
best_model_number = None

# Keep the results so we can make graphs later.
model_results = []

for model_number, model in enumerate(models, start=1):
    print(f"\nTraining Model {model_number}...")

    model.fit(X_train_selected, y_train)

    train_pred = model.predict(X_train_selected)
    val_pred = model.predict(X_val_selected)

    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    # SVC does not have the usual neural-network training loss.
    # Log loss gives us a useful train-vs-validation loss comparison.
    train_prob = model.predict_proba(X_train_selected)
    val_prob = model.predict_proba(X_val_selected)

    train_loss = log_loss(y_train, train_prob, labels=[0, 1, 2])
    val_loss = log_loss(y_val, val_prob, labels=[0, 1, 2])

    model_results.append(
        {
            "model": f"Model {model_number}",
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )

    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model
        best_model_number = model_number


# ----------------------------
# Model comparison graphs
# ----------------------------

results_df = pd.DataFrame(model_results)

# 1. Training vs validation accuracy
plt.figure(figsize=(8, 5))
x = np.arange(len(results_df))
width = 0.35

plt.bar(
    x - width / 2,
    results_df["train_accuracy"] * 100,
    width,
    label="Training",
)
plt.bar(
    x + width / 2,
    results_df["val_accuracy"] * 100,
    width,
    label="Validation",
)

plt.xticks(x, results_df["model"])
plt.ylabel("Accuracy (%)")
plt.xlabel("SVM model")
plt.title("Training vs Validation Accuracy")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_vs_validation_accuracy.png", dpi=200)
plt.show()


# 2. Training vs validation loss
plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    results_df["train_loss"],
    width,
    label="Training loss",
)
plt.bar(
    x + width / 2,
    results_df["val_loss"],
    width,
    label="Validation loss",
)

plt.xticks(x, results_df["model"])
plt.ylabel("Log loss")
plt.xlabel("SVM model")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_vs_validation_loss.png", dpi=200)
plt.show()


# 3. Validation accuracy of all models
plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["val_accuracy"] * 100)

for i, value in enumerate(results_df["val_accuracy"] * 100):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.ylabel("Validation accuracy (%)")
plt.xlabel("SVM model")
plt.title("Validation Accuracy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_accuracy_comparison.png", dpi=200)
plt.show()


# Save the model comparison numbers as a CSV.
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

# A large train/validation gap can be a warning sign of overfitting.
results_df["accuracy_gap"] = (
    results_df["train_accuracy"] - results_df["val_accuracy"]
)

print("\nOverfitting check:")
for _, row in results_df.iterrows():
    gap = row["accuracy_gap"] * 100
    print(
        f"{row['model']}: "
        f"train/validation gap = {gap:.2f} percentage points"
    )


# ----------------------------
# Final evaluation
# ----------------------------

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected model:", best_model_number)
print(f"Validation accuracy: {best_val_accuracy * 100:.2f}%")


print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

test_pred = best_model.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%")


# 4. Final accuracy summary
plt.figure(figsize=(7, 5))

accuracy_labels = ["Training", "Validation", "Test"]
accuracy_values = [
    accuracy_score(y_train, best_model.predict(X_train_selected)) * 100,
    best_val_accuracy * 100,
    test_accuracy * 100,
]

plt.bar(accuracy_labels, accuracy_values)

for i, value in enumerate(accuracy_values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.ylabel("Accuracy (%)")
plt.title("Best Model: Training vs Validation vs Test")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_accuracy_summary.png", dpi=200)
plt.show()


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1, 2],
        target_names=SENTIMENT_ORDER,
        zero_division=0,
    )
)


# ----------------------------
# Confusion matrix
# ----------------------------

cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("3-Class Sentiment Confusion Matrix")
plt.colorbar()

plt.xticks(range(3), SENTIMENT_ORDER)
plt.yticks(range(3), SENTIMENT_ORDER)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
)
plt.show()


# ----------------------------
# Per-class precision, recall, F1
# ----------------------------

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    test_pred,
    labels=[0, 1, 2],
    zero_division=0,
)

print("\nPer-class performance:")

for i in range(3):
    print(f"\n{INT_TO_SENTIMENT[i]}")
    print(f"Precision: {precision[i] * 100:.2f}%")
    print(f"Recall   : {recall[i] * 100:.2f}%")
    print(f"F1 Score : {f1[i] * 100:.2f}%")


# ----------------------------
# Final summary
# ----------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nBest validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Final test accuracy: {test_accuracy * 100:.2f}%")

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

print("Original features:", X_train.shape[1])
print("Selected features:", X_train_selected.shape[1])

print("\nSentiment mapping:")
print("Positive = Calm + Happy + Surprised")
print("Negative = Sad + Angry + Fearful + Disgust")
print("Neutral  = Neutral")

print("\nTraining completed successfully!")
