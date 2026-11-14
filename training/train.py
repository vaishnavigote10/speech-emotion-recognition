"""
Train the speech-emotion model on RAVDESS and/or TESS and export it for the web page.

Usage (either dataset can be omitted, but at least one is needed):
    pip install -r requirements.txt
    python train.py --ravdess ../data/Audio_Speech_Actors_01-24 --tess "../data/TESS Toronto emotional speech set data"

Held-out speakers (never used for training, reported separately):
  - RAVDESS actors 21-24
  - TESS speaker YAF (TESS has two speakers: OAF and YAF)

Writes ../models/emotion/ (model.json, weight shards, norm.json).
Features are computed exactly as in index.html (featuresFromFrame),
so the browser sees the same numbers the model was trained on.
"""
import argparse, glob, json, os
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflowjs.converters import save_keras_model
from sklearn.metrics import confusion_matrix

EM = ['Neutral', 'Happy', 'Angry', 'Sad', 'Fearful']
RAVDESS_CODE = {'01': 0, '02': 0, '03': 1, '04': 3, '05': 2, '06': 4}  # calm -> Neutral
TESS_CODE = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4}  # disgust, ps dropped
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

def frames_from(path, idx_cache):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr not in idx_cache:
        idx_cache[sr] = band_index(sr)
    out = []
    for i in range(0, len(x) - N + 1, N):
        f = feats(x[i:i + N], sr, idx_cache[sr])
        if f[0] >= 0.008:      # skip silence, same threshold as the web page
            out.append(f)
    return out

def load_ravdess(root, idx_cache, rows):
    for path in sorted(glob.glob(os.path.join(root, 'Actor_*', '*.wav'))):
        parts = os.path.basename(path)[:-4].split('-')
        if parts[2] not in RAVDESS_CODE:
            continue
        actor = int(parts[6])
        for f in frames_from(path, idx_cache):
            rows.append((f, RAVDESS_CODE[parts[2]], actor >= 21, 'RAVDESS'))

def load_tess(root, idx_cache, rows):
    for path in sorted(glob.glob(os.path.join(root, '**', '*.wav'), recursive=True)):
        name = os.path.basename(path)[:-4].split('_')
        emo = name[-1].lower()
        if emo not in TESS_CODE:
            continue
        held_out = name[0].upper() == 'YAF'
        for f in frames_from(path, idx_cache):
            rows.append((f, TESS_CODE[emo], held_out, 'TESS'))

def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((9,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(EM), activation='softmax'),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def main(args):
    rows, idx_cache = [], {}
    if args.ravdess:
        load_ravdess(args.ravdess, idx_cache, rows)
    if args.tess:
        load_tess(args.tess, idx_cache, rows)
    if not rows:
        raise SystemExit('No audio found. Check the --ravdess and --tess folder paths.')

    X = np.array([r[0] for r in rows], dtype=np.float32)
    Y = np.array([r[1] for r in rows])
    T = np.array([r[2] for r in rows])
    S = np.array([r[3] for r in rows])
    for src in sorted(set(S)):
        print(f'{src}: {int((S == src).sum())} voiced frames')
    if T.sum() == 0 or (~T).sum() == 0:
        raise SystemExit('Need both training and held-out speakers. Use both datasets.')

    train = ~T
    mu = X[train].mean(0)
    sg = X[train].std(0) + 1e-8
    Xn = (X - mu) / sg

    model = build_model()
    model.fit(Xn[train], Y[train], validation_data=(Xn[T], Y[T]),
              epochs=60, batch_size=128, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

    pred = model.predict(Xn[T], verbose=0).argmax(1)
    acc = float((pred == Y[T]).mean())
    print('Held-out speaker accuracy (all sources, frame level):', round(acc, 4))
    for src in sorted(set(S[T])):
        m = S[T] == src
        print(f'  {src} held-out accuracy: {(pred[m] == Y[T][m]).mean():.4f}')
    print('Confusion matrix (rows = actual, cols = predicted):')
    print(confusion_matrix(Y[T], pred, labels=range(len(EM))))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'emotion')
    os.makedirs(out, exist_ok=True)
    save_keras_model(model, out)
    with open(os.path.join(out, 'norm.json'), 'w') as fh:
        json.dump({'mu': mu.tolist(), 'sg': sg.tolist(), 'test_acc': acc,
                   'labels': EM, 'sources': sorted(set(S))}, fh)
    print('Saved model to', os.path.abspath(out))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ravdess', help='RAVDESS folder containing Actor_01 ... Actor_24')
    ap.add_argument('--tess', help='TESS folder containing OAF_* and YAF_* subfolders')
    main(ap.parse_args())
