import pyaudio
import wave
import numpy as np
import noisereduce as nr
import speech_recognition as sr
from googletrans import Translator

# 配置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 0.01  # 静音阈值
MIN_SILENCE_DURATION = 3.0  # 最小静音持续时间（秒）
RECORD_SECONDS = 15  # 每次录音的持续时间

# 初始化
audio = pyaudio.PyAudio()
recognizer = sr.Recognizer()
translator = Translator()

def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    return b''.join(frames)

def reduce_noise(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    reduced_noise = nr.reduce_noise(y=audio_np, sr=RATE)
    return reduced_noise.astype(np.int16).tobytes()

def recognize_speech(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    audio_source = sr.AudioData(audio_np.tobytes(), RATE, 2)
    try:
        text = recognizer.recognize_google(audio_source)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

def translate_text(text, dest_language='en'):
    translation = translator.translate(text, dest=dest_language)
    print(f"Translated: {translation.text}")
    return translation.text

def main():
    while True:
        audio_data = record_audio()
        reduced_noise_audio = reduce_noise(audio_data)
        text = recognize_speech(reduced_noise_audio)
        if text:
            translate_text(text, dest_language='en')  # 翻译成英文

if __name__ == "__main__":
    main()