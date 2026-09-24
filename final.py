# ============================================================
# SPEECH SENTIMENT RECOGNITION
# 3-CLASS: POSITIVE / NEGATIVE / NEUTRAL
#
# IMPROVED VERSION
# - Actor-independent split
# - Strong audio features
# - Delta + Delta-Delta
# - Pitch features
# - Audio augmentation
# - Class balancing
# - Feature selection
# - Standardization
# - RBF SVM
# - Validation-based model selection
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "/Users/macbook/Downloads/Audio_Speech_Actors_01-24_16k"

OUTPUT_DIR = "outputs_best_sentiment"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. ACTOR-INDEPENDENT SPLIT
# ============================================================

TRAIN_ACTORS = [f"{i:02d}" for i in range(1, 17)]

VAL_ACTORS = [f"{i:02d}" for i in range(17, 21)]

TEST_ACTORS = [f"{i:02d}" for i in range(21, 25)]


# ============================================================
# 3. RAVDESS EMOTION MAPPING
# ============================================================

EMOTION_NAMES = {

    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"

}


# ============================================================
# 4. CONVERT 8 EMOTIONS INTO 3 SENTIMENTS
# ============================================================

SENTIMENT_MAP = {

    "Neutral": "Neutral",

    "Calm": "Positive",
    "Happy": "Positive",
    "Surprised": "Positive",

    "Sad": "Negative",
    "Angry": "Negative",
    "Fearful": "Negative",
    "Disgust": "Negative"

}


SENTIMENT_TO_INT = {

    "Negative": 0,
    "Neutral": 1,
    "Positive": 2

}


INT_TO_SENTIMENT = {

    0: "Negative",
    1: "Neutral",
    2: "Positive"

}


SENTIMENT_ORDER = [
    "Negative",
    "Neutral",
    "Positive"
]


# ============================================================
# 5. CHECK DATASET
# ============================================================

print("=" * 70)

print("SPEECH SENTIMENT RECOGNITION")

print("3-CLASS POSITIVE / NEGATIVE / NEUTRAL")

print("=" * 70)


if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


print("\nDataset found successfully!")

print(DATASET_PATH)


# ============================================================
# 6. READ RAVDESS INFORMATION
# ============================================================

def get_emotion_code(filepath):

    filename = os.path.basename(filepath)

    parts = filename.replace(
        ".wav",
        ""
    ).split("-")

    if len(parts) < 7:
        return None

    return parts[2]


def get_emotion(filepath):

    code = get_emotion_code(filepath)

    if code not in EMOTION_NAMES:
        return None

    return EMOTION_NAMES[code]


def get_sentiment(filepath):

    emotion = get_emotion(filepath)

    if emotion is None:
        return None

    return SENTIMENT_MAP[emotion]


def get_actor(filepath):

    filename = os.path.basename(filepath)

    parts = filename.replace(
        ".wav",
        ""
    ).split("-")

    if len(parts) < 7:
        return None

    return parts[-1]


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

records = []


for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.lower().endswith(".wav"):
            continue

        filepath = os.path.join(
            root,
            file
        )

        emotion = get_emotion(filepath)

        sentiment = get_sentiment(filepath)

        actor = get_actor(filepath)

        if (
            emotion is not None
            and sentiment is not None
            and actor is not None
        ):

            records.append({

                "path": filepath,

                "emotion": emotion,

                "sentiment": sentiment,

                "actor": actor

            })


df = pd.DataFrame(records)


print("\nTotal valid audio files:", len(df))


# ============================================================
# 8. ACTOR-INDEPENDENT SPLIT
# ============================================================

train_df = df[
    df["actor"].isin(TRAIN_ACTORS)
].copy()


val_df = df[
    df["actor"].isin(VAL_ACTORS)
].copy()


test_df = df[
    df["actor"].isin(TEST_ACTORS)
].copy()


print("\n" + "=" * 70)

print("ACTOR-INDEPENDENT SPLIT")

print("=" * 70)

print("\nTraining actors:")
print(TRAIN_ACTORS)

print("\nValidation actors:")
print(VAL_ACTORS)

print("\nTest actors:")
print(TEST_ACTORS)


print("\nTraining files   :", len(train_df))
print("Validation files :", len(val_df))
print("Testing files   :", len(test_df))


# ============================================================
# 9. PRINT SENTIMENT DISTRIBUTION
# ============================================================

def show_distribution(name, dataframe):

    print("\n" + name)

    print("-" * 40)

    counts = dataframe["sentiment"].value_counts()

    for sentiment in SENTIMENT_ORDER:

        print(
            f"{sentiment:<10}: "
            f"{counts.get(sentiment, 0)}"
        )


show_distribution(
    "TRAINING DISTRIBUTION",
    train_df
)

show_distribution(
    "VALIDATION DISTRIBUTION",
    val_df
)

show_distribution(
    "TEST DISTRIBUTION",
    test_df
)


# ============================================================
# 10. AUDIO AUGMENTATION
# ============================================================

def add_noise(y):

    noise = np.random.normal(
        0,
        0.003,
        len(y)
    )

    return y + noise


def pitch_shift(y, sr):

    steps = np.random.uniform(
        -1.5,
        1.5
    )

    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=steps
    )


def time_stretch(y):

    rate = np.random.uniform(
        0.90,
        1.10
    )

    return librosa.effects.time_stretch(
        y=y,
        rate=rate
    )


# ============================================================
# 11. FEATURE EXTRACTION
# ============================================================

def extract_features_from_signal(
    y,
    sr
):

    # --------------------------------------------------------
    # Remove silence
    # --------------------------------------------------------

    y, _ = librosa.effects.trim(
        y,
        top_db=25
    )

    if len(y) < 1000:

        return None


    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )


    features = []


    # MFCC statistics

    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )

    features.extend(
        np.mean(mfcc_delta, axis=1)
    )

    features.extend(
        np.std(mfcc_delta, axis=1)
    )

    features.extend(
        np.mean(mfcc_delta2, axis=1)
    )

    features.extend(
        np.std(mfcc_delta2, axis=1)
    )


    # ========================================================
    # CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(chroma, axis=1)
    )

    features.extend(
        np.std(chroma, axis=1)
    )


    # ========================================================
    # MEL SPECTROGRAM
    # ========================================================

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    features.extend(
        np.mean(mel_db, axis=1)
    )

    features.extend(
        np.std(mel_db, axis=1)
    )


    # ========================================================
    # SPECTRAL CONTRAST
    # ========================================================

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    features.extend(
        np.mean(contrast, axis=1)
    )

    features.extend(
        np.std(contrast, axis=1)
    )


    # ========================================================
    # ZERO CROSSING RATE
    # ========================================================

    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    features.append(
        np.mean(zcr)
    )

    features.append(
        np.std(zcr)
    )


    # ========================================================
    # RMS ENERGY
    # ========================================================

    rms = librosa.feature.rms(
        y=y
    )

    features.append(
        np.mean(rms)
    )

    features.append(
        np.std(rms)
    )


    # ========================================================
    # SPECTRAL CENTROID
    # ========================================================

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(centroid)
    )

    features.append(
        np.std(centroid)
    )


    # ========================================================
    # SPECTRAL BANDWIDTH
    # ========================================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(bandwidth)
    )

    features.append(
        np.std(bandwidth)
    )


    # ========================================================
    # SPECTRAL ROLLOFF
    # ========================================================

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(rolloff)
    )

    features.append(
        np.std(rolloff)
    )


    # ========================================================
    # PITCH / F0
    # ========================================================

    try:

        f0, voiced_flag, voiced_prob = (
            librosa.pyin(
                y,
                fmin=librosa.note_to_hz(
                    "C2"
                ),
                fmax=librosa.note_to_hz(
                    "C7"
                ),
                sr=sr
            )
        )

        f0 = f0[
            ~np.isnan(f0)
        ]

        if len(f0) > 0:

            features.extend([

                np.mean(f0),

                np.std(f0),

                np.min(f0),

                np.max(f0),

                np.median(f0)

            ])

        else:

            features.extend(
                [0, 0, 0, 0, 0]
            )

    except:

        features.extend(
            [0, 0, 0, 0, 0]
        )


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# 12. LOAD AUDIO
# ============================================================

def load_audio(filepath):

    y, sr = librosa.load(
        filepath,
        sr=16000,
        mono=True
    )

    return y, sr


# ============================================================
# 13. EXTRACT NORMAL DATA
# ============================================================

def extract_normal_dataset(dataframe):

    X = []

    y = []

    for i, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        try:

            signal, sr = load_audio(
                row["path"]
            )

            features = extract_features_from_signal(
                signal,
                sr
            )

            if features is not None:

                X.append(features)

                y.append(
                    SENTIMENT_TO_INT[
                        row["sentiment"]
                    ]
                )

        except Exception as e:

            print(
                "Error:",
                row["path"],
                e
            )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(dataframe)}"
            )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 14. TRAINING DATA WITH AUGMENTATION
# ============================================================

def extract_training_dataset(dataframe):

    X = []

    y = []

    for i, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        try:

            signal, sr = load_audio(
                row["path"]
            )

            label = SENTIMENT_TO_INT[
                row["sentiment"]
            ]


            # ------------------------------------------------
            # ORIGINAL
            # ------------------------------------------------

            features = extract_features_from_signal(
                signal,
                sr
            )

            if features is not None:

                X.append(features)
                y.append(label)


            # ------------------------------------------------
            # NOISE
            # ------------------------------------------------

            noisy = add_noise(
                signal
            )

            features = extract_features_from_signal(
                noisy,
                sr
            )

            if features is not None:

                X.append(features)
                y.append(label)


            # ------------------------------------------------
            # PITCH
            # ------------------------------------------------

            pitched = pitch_shift(
                signal,
                sr
            )

            features = extract_features_from_signal(
                pitched,
                sr
            )

            if features is not None:

                X.append(features)
                y.append(label)


            # ------------------------------------------------
            # TIME STRETCH
            # ------------------------------------------------

            stretched = time_stretch(
                signal
            )

            features = extract_features_from_signal(
                stretched,
                sr
            )

            if features is not None:

                X.append(features)
                y.append(label)


        except Exception as e:

            print(
                "Training error:",
                row["path"],
                e
            )


        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(dataframe)}"
            )


    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 15. EXTRACT TRAINING FEATURES
# ============================================================

print("\n" + "=" * 70)

print("EXTRACTING TRAINING DATA")

print("=" * 70)


X_train, y_train = extract_training_dataset(
    train_df
)


# ============================================================
# 16. VALIDATION
# ============================================================

print("\n" + "=" * 70)

print("EXTRACTING VALIDATION DATA")

print("=" * 70)


X_val, y_val = extract_normal_dataset(
    val_df
)


# ============================================================
# 17. TEST
# ============================================================

print("\n" + "=" * 70)

print("EXTRACTING TEST DATA")

print("=" * 70)


X_test, y_test = extract_normal_dataset(
    test_df
)


print("\nFeature extraction complete!")

print("Training:", X_train.shape)

print("Validation:", X_val.shape)

print("Test:", X_test.shape)


# ============================================================
# 18. HANDLE INVALID VALUES
# ============================================================

X_train = np.nan_to_num(
    X_train,
    nan=0,
    posinf=0,
    neginf=0
)

X_val = np.nan_to_num(
    X_val,
    nan=0,
    posinf=0,
    neginf=0
)

X_test = np.nan_to_num(
    X_test,
    nan=0,
    posinf=0,
    neginf=0
)


# ============================================================
# 19. SCALE
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 20. FEATURE SELECTION
# ============================================================

print("\nSelecting best features...")

K_FEATURES = min(
    300,
    X_train_scaled.shape[1]
)


selector = SelectKBest(
    score_func=mutual_info_classif,
    k=K_FEATURES
)


X_train_selected = selector.fit_transform(
    X_train_scaled,
    y_train
)


X_val_selected = selector.transform(
    X_val_scaled
)


X_test_selected = selector.transform(
    X_test_scaled
)


print(
    "Original features:",
    X_train.shape[1]
)

print(
    "Selected features:",
    X_train_selected.shape[1]
)


# ============================================================
# 21. BALANCE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)

print("TRAINING CLASS DISTRIBUTION")

print("=" * 70)


counts = Counter(y_train)

for class_id in range(3):

    print(
        INT_TO_SENTIMENT[class_id],
        ":",
        counts[class_id]
    )


# ============================================================
# 22. TRAIN MULTIPLE SVM MODELS
# ============================================================

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
        random_state=42
    ),

    SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ),

    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ),

    SVC(
        kernel="rbf",
        C=5,
        gamma=0.01,
        class_weight="balanced",
        probability=True,
        random_state=42
    )

]


best_model = None

best_val_accuracy = 0

best_model_number = 0


for i, svm_model in enumerate(
    models,
    start=1
):

    print(
        f"\nTraining Model {i}..."
    )

    svm_model.fit(
        X_train_selected,
        y_train
    )


    val_pred = svm_model.predict(
        X_val_selected
    )


    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )


    print(
        f"Model {i} Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model = svm_model

        best_model_number = i


# ============================================================
# 23. BEST MODEL
# ============================================================

print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)


print(
    "Selected Model:",
    best_model_number
)

print(
    f"Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 24. FINAL TEST
# ============================================================

print("\n" + "=" * 70)

print("FINAL TEST")

print("=" * 70)


test_pred = best_model.predict(
    X_test_selected
)


test_accuracy = accuracy_score(
    y_test,
    test_pred
)


print(
    f"\nFINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)


print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1, 2],
        target_names=SENTIMENT_ORDER,
        zero_division=0
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1, 2]
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 27. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "3-Class Sentiment Confusion Matrix"
)


plt.colorbar()


plt.xticks(
    range(3),
    SENTIMENT_ORDER
)


plt.yticks(
    range(3),
    SENTIMENT_ORDER
)


plt.xlabel(
    "Predicted Sentiment"
)


plt.ylabel(
    "Actual Sentiment"
)


for i in range(3):

    for j in range(3):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)


plt.show()


# ============================================================
# 28. PRECISION RECALL F1
# ============================================================

precision, recall, f1, support = (

    precision_recall_fscore_support(

        y_test,
        test_pred,

        labels=[0, 1, 2],

        zero_division=0

    )

)


print("\nPer-class performance:")

for i in range(3):

    print(
        f"\n{INT_TO_SENTIMENT[i]}"
    )

    print(
        f"Precision: {precision[i] * 100:.2f}%"
    )

    print(
        f"Recall   : {recall[i] * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1[i] * 100:.2f}%"
    )


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("FINAL SUMMARY")

print("=" * 70)


print(
    "\nBest validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    "Final test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Validation samples:",
    len(X_val)
)


print(
    "Test samples:",
    len(X_test)
)


print(
    "Original features:",
    X_train.shape[1]
)


print(
    "Selected features:",
    X_train_selected.shape[1]
)


print("\nSentiment mapping:")

print(
    "Positive = Calm + Happy + Surprised"
)

print(
    "Negative = Sad + Angry + Fearful + Disgust"
)

print(
    "Neutral = Neutral"
)


print("\nTraining completed successfully!")