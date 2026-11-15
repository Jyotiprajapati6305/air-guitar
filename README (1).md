# Air Guitar

Play guitar in the air with your webcam. Your **left hand** picks the chord (number of raised fingers) and your **right hand** strums (sweep down or up). Built with Python, OpenCV, MediaPipe and pygame.

## Run it (Windows, with a webcam)

Requires Python 3.9 - 3.12.

```
git clone https://github.com/Jyotiprajapati6305/air-guitar.git
cd air-guitar
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Press **Q** or **ESC** to quit.

## Chords (left hand)

| Fingers up | Chord |
|---|---|
| 0 (fist) | Em |
| 1 | G |
| 2 | C |
| 3 | D |
| 4 | Am |
| 5 | A |

Keys **1-6** lock a chord, **0** returns to hand control.

## Strumming model

`strum_model.npz` is a small pre-trained model (trained on simulated strums). To fit it to your own style:

```
python record_strums.py       # record DOWN, UP and IDLE sessions
python train_strum_model.py   # creates a new strum_model.npz
```

`python make_default_model.py` regenerates the default model. Settings live in `config.py`.

## Tests

```
pip install pytest scikit-learn
pytest
```
