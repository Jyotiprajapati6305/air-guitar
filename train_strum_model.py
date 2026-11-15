"""Train the strum classifier from strum_data.csv (made by record_strums.py)."""
import numpy as np
import config


def train(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = RandomForestClassifier(n_estimators=30, class_weight="balanced", random_state=0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = float((pred == yte).mean())
    return model, acc, confusion_matrix(yte, pred)


def main():
    import joblib
    data = np.loadtxt(config.DATA_PATH, delimiter=",", dtype=np.float32)
    X, y = data[:, :-1], data[:, -1].astype(int)
    counts = np.bincount(y, minlength=3)
    print(f"Samples  idle:{counts[0]}  down:{counts[1]}  up:{counts[2]}")
    if counts[1] < 30 or counts[2] < 30 or counts[0] < 100:
        print("Not enough stroke examples. Record more DOWN / UP / IDLE sessions first.")
        return 1
    model, acc, cm = train(X, y)
    print(f"Held-out accuracy: {acc:.1%}\nConfusion matrix (rows=true idle/down/up):\n{cm}")
    joblib.dump(model, config.MODEL_PATH)
    print(f"Saved {config.MODEL_PATH}. main.py will use it automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
