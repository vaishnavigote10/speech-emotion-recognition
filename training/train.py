"""
Train the speech-emotion model on RAVDESS and export it for the web page.

Usage:
    pip install -r requirements.txt
    python train.py path/to/Audio_Speech_Actors_01-24

Writes ../models/emotion/ (model.json, weight shards, norm.json).
Features are computed exactly as in index.html (featuresFromFrame),
so the browser sees the same numbers the model was trained on.
"""
import glob, json, os, sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflowjs.converters import save_keras_model
from sklearn.metrics import confusion_matrix

EM = ['Neutral', 'Happy', 'Angry', 'Sad', 'Fearful']
# RAVDESS emotion codes -> our labels (calm merges into Neutral; others are dropped)
CODE = {'01': 0, '02': 0, '03': 1, '04': 3, '05': 2, '06': 4}
N = 2048
W = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)   # periodic Hann, same as JS

def mel(f):
    return 2595 * np.log10(1 + f / 700)

def band_index(sr):
    k = np.arange(N // 2)
    f = k * sr / N
    lo, hi = mel(100.0), mel(4000.0)
    idx = np.full(N // 2, -1)
    ok = (k >= 1) & (f >= 100) & (f <= 4000)
    idx[ok] = np.minimum(19, np.floor((mel(f[ok]) - lo) / (hi - lo) * 20)).astype(int)
    return idx

def feats(x, sr, idx):
    e = float(np.dot(x, x))
    rms = np.sqrt(e / N)
    zcr = np.sum((x[1:] >= 0) != (x[:-1] >= 0)) / N
    best, bl = 0.0, 0
    for L in range(int(sr // 400), int(sr // 70) + 1):
        s = np.dot(x[:N - L], x[L:]) / (e if e > 0 else 1.0)
        if s > best:
            best, bl = s, L
    pitch = sr / bl if (rms > 0.01 and best > 0.3 and bl) else 0.0
    P = np.abs(np.fft.rfft(x * W)) ** 2
    p = P[1:N // 2]
    f = np.arange(1, N // 2) * sr / N
    tot = p.sum()
    centroid = (f * p).sum() / tot if tot > 0 else 0.0
    E = np.zeros(20)
    m = idx[1:N // 2]
    v = m >= 0
    np.add.at(E, m[v], p[v])
    logE = np.log(E + 1e-10)
    b = np.arange(20)
    mf = [float(np.sum(logE * np.cos(np.pi * k * (b + 0.5) / 20))) for k in range(1, 6)]
    return [rms, zcr, float(pitch), float(centroid)] + mf

def load(root):
    X, y, actors = [], [], []
    idx_cache = {}
    for path in sorted(glob.glob(os.path.join(root, 'Actor_*', '*.wav'))):
        parts = os.path.basename(path)[:-4].split('-')
        code = parts[2]
        if code not in CODE:
            continue
        x, sr = sf.read(path, dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr not in idx_cache:
            idx_cache[sr] = band_index(sr)
        actor = int(parts[6])
        for i in range(0, len(x) - N + 1, N):
            f = feats(x[i:i + N], sr, idx_cache[sr])
            if f[0] < 0.008:
                continue
            X.append(f)
            y.append(CODE[code])
            actors.append(actor)
    return np.array(X), np.array(y), np.array(actors)

def main(root):
    X, y, actors = load(root)
    print(f'Loaded {len(X)} voiced frames from {len(set(actors))} speakers')
    test = actors >= 21          # speaker-independent split: actors 21-24 held out
    train = ~test
    mu = X[train].mean(0)
    sg = X[train].std(0) + 1e-8
    Xn = (X - mu) / sg

    model = tf.keras.Sequential([
        tf.keras.layers.Input((9,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(EM), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(Xn[train], y[train], validation_data=(Xn[test], y[test]),
              epochs=60, batch_size=128, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

    pred = model.predict(Xn[test], verbose=0).argmax(1)
    acc = float((pred == y[test]).mean())
    print('Held-out speaker accuracy (frame level):', round(acc, 4))
    print(confusion_matrix(y[test], pred, labels=range(len(EM))))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'emotion')
    os.makedirs(out, exist_ok=True)
    save_keras_model(model, out)
    with open(os.path.join(out, 'norm.json'), 'w') as fh:
        json.dump({'mu': mu.tolist(), 'sg': sg.tolist(), 'test_acc': acc, 'labels': EM}, fh)
    print('Saved model to', os.path.abspath(out))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python train.py <RAVDESS root folder>')
    main(sys.argv[1])
