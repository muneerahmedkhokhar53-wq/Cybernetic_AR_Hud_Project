"""Collect hand-landmark samples and train a custom gesture classifier.

  1) python train_gesture.py collect --labels open_palm fist point peace thumbs_up rock
       - press 1..9 to choose the label, SPACE to start/stop recording samples, Q to save + quit
       - record both hands and a few different angles/distances for each gesture (200+ samples each)
  2) python train_gesture.py train
       - trains a small neural network and writes models/gesture_model.pkl

The HUD loads that model automatically (gestures.use_model in config.yaml). Keep the label
names open_palm / fist / point / peace if you want modes that rely on them to keep working
(swipes need OPEN_PALM, DRAW needs POINT, HOLO needs PEACE).
"""
import argparse
import os
import pickle

import cv2
import numpy as np

from hud.gestures import landmark_features

DATA_PATH = "data/gesture_samples.npz"
MODEL_PATH = "models/gesture_model.pkl"


def collect(labels, camera_index):
    import mediapipe as mp

    os.makedirs("data", exist_ok=True)
    X, y = [], []
    if os.path.exists(DATA_PATH):
        old = np.load(DATA_PATH, allow_pickle=True)
        X, y = list(old["X"]), list(old["y"])
        print(f"Loaded {len(y)} existing samples")

    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(camera_index)
    current, recording = 0, False
    print("Keys: 1-9 select label | SPACE record on/off | Q save + quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                       # same mirroring as the HUD
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
            if recording:
                X.append(landmark_features(arr))
                y.append(labels[current])
        counts = {lab: y.count(lab) for lab in labels}
        cv2.putText(frame, f"LABEL: {labels[current]}  {'[REC]' if recording else '[paused]'}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if recording else (0, 255, 0), 2)
        for i, lab in enumerate(labels):
            cv2.putText(frame, f"{i + 1}: {lab} ({counts[lab]})", (20, 70 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("collect gestures", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            recording = not recording
        if ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
            current = key - ord("1")
    cap.release()
    cv2.destroyAllWindows()
    if y:
        np.savez(DATA_PATH, X=np.array(X), y=np.array(y))
        print(f"Saved {len(y)} samples to {DATA_PATH}")


def train():
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = np.load(DATA_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]
    # augmentation: mirror x so left- and right-hand samples both help
    flipped = X.reshape(len(X), 21, 3).copy()
    flipped[:, :, 0] *= -1
    X = np.vstack([X, flipped.reshape(len(X), -1)])
    y = np.concatenate([y, y])
    X = X + np.random.normal(0, 0.01, X.shape)           # small jitter

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=0))
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te)))
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model}, fh)
    print(f"Model saved to {MODEL_PATH} - classes: {list(model.classes_)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--labels", nargs="+", default=["open_palm", "fist", "point", "peace", "thumbs_up", "rock"])
    c.add_argument("--camera", type=int, default=0)
    sub.add_parser("train")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.labels, args.camera)
    else:
        train()
