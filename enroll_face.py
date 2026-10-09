"""Enroll a face for the HUD's optional face identification.

    python enroll_face.py --name Ali --samples 8

Only enroll people who have given their consent. Encodings (not photos) are stored locally in
faces/encodings.pkl; delete that file to forget everyone. Requires: pip install face_recognition
"""
import argparse
import os
import pickle
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dir", default="faces")
    args = ap.parse_args()

    import face_recognition

    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, "encodings.pkl")
    data = {"names": [], "encodings": []}
    if os.path.exists(path):
        with open(path, "rb") as fh:
            data = pickle.load(fh)

    cap = cv2.VideoCapture(args.camera)
    got, last = 0, 0.0
    print(f"Look at the camera and slowly turn your head. Capturing {args.samples} samples. Q to abort.")
    while got < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if len(boxes) == 1 and time.time() - last > 0.6:
            enc = face_recognition.face_encodings(rgb, boxes)[0]
            data["names"].append(args.name)
            data["encodings"].append(enc)
            got += 1
            last = time.time()
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{args.name}: {got}/{args.samples}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("enroll face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"Saved. {len(data['names'])} encodings on file in {path}")


if __name__ == "__main__":
    main()
