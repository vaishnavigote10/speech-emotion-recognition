# Emotion Recognition from Speech

A speech emotion dashboard. Upload any audio file (MP3, WAV, OGG, M4A, FLAC) or use the microphone. The app extracts acoustic features (RMS energy, zero-crossing rate, pitch, spectral centroid, MFCC-style coefficients) and classifies them as Neutral, Happy, Angry, Sad or Fearful with a neural network running in TensorFlow.js.

## Run it

Serve over HTTPS (GitHub Pages) or localhost. Microphone access does not work in embedded previews, so open the page in its own tab.

- GitHub Pages: Settings → Pages → Deploy from a branch → `main` / root.
- Local: `python -m http.server 8000`, then open `http://localhost:8000`.

## Microphone permission

Click the site-settings icon in the address bar, set Microphone to Allow, reload, and press Start again. On Windows, check Settings → Privacy & security → Microphone.

## Train the model on RAVDESS and TESS

Datasets are not stored in this repository. Download them into a local `data/` folder.

1. Download **RAVDESS** (Audio_Speech_Actors_01-24.zip) from Zenodo: search "RAVDESS Zenodo".
2. Download **TESS** from Kaggle: search "Toronto emotional speech set (TESS)".
3. Extract both into `data/`. You should have:
   ```
   data/Audio_Speech_Actors_01-24/Actor_01 ... Actor_24
   data/TESS Toronto emotional speech set data/OAF_angry ... YAF_sad
   ```
4. From the `training/` folder:
   ```bash
   pip install -r requirements.txt
   python train.py --ravdess ../data/Audio_Speech_Actors_01-24 --tess "../data/TESS Toronto emotional speech set data"
   ```
   Either flag can be dropped to train on one dataset only.
5. Commit the generated `models/emotion/` folder. The page loads it automatically and shows the held-out accuracy.

Held-out speakers are never used for training: RAVDESS actors 21–24 and TESS speaker YAF. The script reports accuracy for each dataset separately, so the numbers reflect speakers the model has not heard.

## Limitations

- No model reaches 100% on speech emotion. Accuracy varies by speaker, language, and recording quality.
- The 9-feature model is deliberately simple. Higher accuracy would need richer features (full MFCC sequences) and a CNN or LSTM.
- TESS contains only two speakers, both female, so the model may generalise less well to men and children.

## License

Add a license of your choice before publishing (for example, MIT).
