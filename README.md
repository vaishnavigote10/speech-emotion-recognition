# Emotion Recognition from Speech

A real-time speech emotion dashboard. It extracts acoustic features (RMS energy, zero-crossing rate, pitch, spectral centroid and MFCC-style coefficients) from the microphone or an audio file, then classifies them as Neutral, Happy, Angry, Sad or Fearful with a neural network built in TensorFlow.js.

## Run it

No build step is needed. The page must be served over HTTPS or localhost for the microphone to work.

1. Clone or download this repository.
2. Open the GitHub Pages link (see below), or run `python -m http.server 8000` and visit `http://localhost:8000`.
3. Click **Start microphone** and allow access when the browser asks.

## Fixing "microphone permission denied"

- Click the site-settings icon (lock or tune icon) in the address bar, set **Microphone** to **Allow**, reload, and press Start again.
- On Windows, check **Settings → Privacy & security → Microphone** and make sure both "Microphone access" and "Let apps access your microphone" are on.
- Embedded previews (for example, inside a chat window or an iframe) often block microphone access. Open the page in its own browser tab.
- If no microphone is available, use **Analyse an audio file** to upload a WAV or MP3 recording.

## Enable GitHub Pages

1. Push this folder to the root of a public repository.
2. Go to **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
3. The site appears at `https://<your-username>.github.io/<repo-name>/`.

## How it works

1. The Web Audio API analyses 2048-sample frames every 200 ms.
2. Features are computed: RMS energy, zero-crossing rate, pitch via autocorrelation (70–400 Hz), spectral centroid, and five MFCC-style coefficients from a 20-band mel filterbank and DCT.
3. Features are standardized and passed to a 2-layer neural network (32 hidden ReLU units, softmax over five emotions).
4. Predictions are smoothed over the last five frames to reduce flicker.

## Limitations

- The classifier is trained on **synthetic** feature distributions that approximate emotional speech. It has not been trained on real recordings such as RAVDESS, TESS or EMO-DB, so live predictions are a demonstration of the pipeline, not a reliable emotion detector.
- Microphone features are not calibrated against real speech.
- No model reaches 100% accuracy. Real speech emotion recognition typically reaches 70–85% with models trained on the proper datasets.

## Next steps

Train a CNN or LSTM on RAVDESS or TESS in Python (using `librosa` for MFCC extraction), export the model to TensorFlow.js, and load it in `index.html` in place of the synthetic classifier.

## License

Add a license of your choice before publishing (for example, MIT).
