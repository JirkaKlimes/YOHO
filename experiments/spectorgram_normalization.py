import matplotlib.pyplot as plt

from yoho.src.preprocessing.audio import mel_spectogram, normalize_spectogram, load_audio

audio = load_audio("./data/audio/sample.mp3", 16000)

spec = mel_spectogram(audio[2 * 16000 : 4 * 16000], 400, 160, 16000, 128)
spec = normalize_spectogram(spec)

plt.imshow(spec.T)

plt.show()
