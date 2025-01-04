# pip install SpeechRecognition
# pip install pyaudio
# pip install requests
# pip install pyttsx3
# pip install gTTS
# pip install playsound
# pip install vosk
# pip install langid


import speech_recognition as sr
# import openai
import requests
import json
import base64

import pyttsx3

from gtts import gTTS
import os
import playsound

import os
import wave
from vosk import Model, KaldiRecognizer # C:\@kunsheng\vosk-model-small-cn-0.22

import pyaudio
import time
import numpy as np
import langid

from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 定义常量
THRESHOLD = 500  # 声音的振幅阈值
SILENCE_LIMIT = 5  # 停止录制的最大静音时间（秒）
RATE = 44100  # 采样率
CHUNK = 1024  # 每次读取的数据块大小
FORMAT = pyaudio.paInt16  # 采样格式
CHANNELS = 1  # 单声道

# 用于保存音频的 WAV 文件
output_filename = "output.wav"

#api_url = "https://api.openai.com/v1/completions"  # 对应 GPT-3 或 GPT-4
api_url = "https://openai-api.agilearch.tech/v1/chat/completions"
# 设置OpenAI API密钥
api_key = os.environ.get("OPENAI_API_KEY")
# 替换为你的API密钥 从 https://platform.openai.com/ 获得
# https://platform.openai.com/settings/organization/api-keys

# 开始录音的函数
def record_audio():
    # 初始化 pyaudio
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    silence_counter = 0  # 记录静音时长（秒）
    
    print("开始录音，停止录音超过 5 秒将自动保存...")
    
    # 循环录音
    while True:
        # 读取数据块
        data = stream.read(CHUNK)
        frames.append(data)
        
        # 将数据转换为 numpy 数组
        audio_data = np.frombuffer(data, dtype=np.int16)

         # 处理无效值（防止无效值干扰计算）
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            continue  # 跳过这个数据块
        
        # 判断是否有声音（通过计算音频的 RMS 值）
        rms = np.sqrt(np.mean(audio_data**2))
        
        # 如果声音大于阈值，重置静音计数器
        if rms > THRESHOLD:
            silence_counter = 0
        else:
            silence_counter += 1
        
        # 如果静音时间超过 SILENCE_LIMIT 秒，停止录音
        if silence_counter > SILENCE_LIMIT * (RATE / CHUNK):
            print("检测到静音超过 {} 秒，停止录音".format(SILENCE_LIMIT))
            break
    
    # 停止音频流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 将音频数据保存为 wav 文件
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"录音已保存为 {output_filename}")

def speech_to_text2():
    # 初始化语音识别器
    recognizer = sr.Recognizer()

    # 使用麦克风作为音频输入源
    with sr.Microphone() as source:
        print("请开始说话...")
        # 调整噪声环境
        recognizer.adjust_for_ambient_noise(source)
        # 录制语音
        audio = recognizer.listen(source)
        print("识别中...")

        try:
            # 使用Google Web语音识别服务将语音转换为文字
            text = recognizer.recognize_google(audio, language='zh-CN')  # 'zh-CN' 代表中文
            return text
        except sr.UnknownValueError:
            return "无法理解音频"
        except sr.RequestError:
            return "无法请求语音识别服务；请检查网络连接"

# 加载中文和英文模型
def get_model(language):
    if language == 'en':
        return Model("C:\\@kunsheng\\vosk-model-small-en-us-0.15")  # 英文模型路径
    elif language == 'zh':
        return Model("C:\\@kunsheng\\vosk-model-small-cn-0.22")  # 中文模型路径
    else:
        raise ValueError(f"Unsupported language: {language}")

def detect_language_with_langid(text):
    lang, _ = langid.classify(text)
    return lang

def speech_to_text(audio_file):
    segments, info = model.transcribe(audio_file, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    text = ""

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text += segment.text
    
    return text

def speech_to_text3(audio_file):
    # 加载 Vosk 中文模型
    #model = Model("C:\\@kunsheng\\vosk-model-small-cn-0.22")  # 替换为模型的实际路径
    
    # 打开音频文件
    wf = wave.open(audio_file, "rb")
    # 获取音频采样率
    sample_rate = wf.getframerate()
    #recognizer = KaldiRecognizer(model, wf.getframerate())

    # 加载英文和中文模型
    model_en = get_model("en")
    model_zh = get_model("zh")
    
    recognizer_en = KaldiRecognizer(model_en, sample_rate)
    recognizer_zh = KaldiRecognizer(model_zh, sample_rate)
    
    frames = wf.readframes(wf.getnframes())
    
    result_en = recognizer_en.AcceptWaveform(frames)
    result_zh = recognizer_zh.AcceptWaveform(frames)
    
    # 识别英文文本
    if result_en:
        result_en_dict = json.loads(recognizer_en.Result())
        print("英文识别结果：", result_en_dict['text'])
    
    # 识别中文文本
    if result_zh:
        result_zh_dict = json.loads(recognizer_zh.Result())
        print("中文识别结果：", result_zh_dict['text'])

    # 通过 langid 检测语言
    chinese_lang = detect_language_with_langid(result_zh_dict['text'])
    english_lang = detect_language_with_langid(result_en_dict['text'])

    print(chinese_lang, english_lang)
    if chinese_lang == 'zh':
        return result_zh_dict['text']
    else:
        return result_en_dict['text']

    #return result
    # 逐帧识别
    #while True:
    #    data = wf.readframes(4000)
    #    if len(data) == 0:
    #        break
    #    if recognizer.AcceptWaveform(data):
    #        result = recognizer.Result()
    #        print("A:")
    #        print(result)
    #        result_dict = json.loads(result)
    #        # 访问识别的文本
    #        print("识别文本：", result_dict["text"])
    #        return result_dict["text"]
    
    # 输出最后的结果
    #print("B:")
    #print(recognizer.FinalResult())

    #result_dict = json.loads(recognizer.FinalResult())
    # 访问识别的文本
    #print("识别文本：", result_dict["text"])
    #return result_dict["text"]

def get_openai_response(prompt):
    try:
        # 调用OpenAI API发送提示并获得回答
        response = openai.Completion.create(
            engine="gpt-4",  # 使用 GPT-4，也可以选择 "gpt-3.5-turbo" 等
            prompt=prompt,
            max_tokens=150,  # 设置返回内容的最大token数
            temperature=0.7,  # 控制回答的创意性，值越高越创意
            n=1,  # 返回的回答数量
            stop=None,  # 设置停止符，可以定义一个特定的停止词
            top_p=1.0,  # 核心采样，保持默认即可
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # 获取生成的文本
        answer = response.choices[0].text.strip()
        return answer

    except Exception as e:
        return f"Error occurred: {str(e)}"

def get_openai_response_via_http(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # 使用 Bearer Token 进行身份验证
    }

    # 请求的数据体
    data = {
        "model": "gpt-4o-mini",  # 可以替换为 "gpt-3.5-turbo" 等
        #"modalities": ["text", "audio"],
        #"audio": { "voice": "alloy", "format": "wav" },
        "messages": [
            {"role": "developer", "content": "你是崔益诚的英语老师Ava"},
            {"role": "user", "content": prompt}
        ],        
        "max_tokens": 150,
        "temperature": 0.7,
        "n": 1
    }

    try:
        # 发送 POST 请求到 OpenAI API
        response = requests.post(api_url, headers=headers, json=data)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 获取响应的 JSON 数据
            response_json = response.json()

            print(response_json)

            # 从 JSON 中提取生成的文本
            # answer = response_json['choices'][0]['text'].strip()

            answer = response_json['choices'][0]['message']['content'].strip()

            # Write audio data to a file
            #wav_bytes = base64.b64decode(response_json['choices'][0]['message']['audio']['data'])
            #with open("dog.wav", "wb") as f:
            #    f.write(wav_bytes)

            return answer
        else:
            return f"请求失败，错误码: {response.status_code}, 错误信息: {response.text}"
    
    except Exception as e:
        return f"请求过程中发生错误: {str(e)}"

def text_to_speech2(text):
    # 初始化 pyttsx3 引擎
    engine = pyttsx3.init()

    # 获取系统支持的语音列表
    voices = engine.getProperty('voices')

    # 打印语音列表的长度
    print(f"语音列表的长度: {len(voices)}")

    # 设置中文语音
    #for voice in voices:
    #    print(voice.languages)
    #    # 检查是否有 'languages' 属性以及是否包含中文（'zh'）
    #    if hasattr(voice, 'languages') and voice.languages:
    #        if "zh" in voice.languages[0]:  # 查找支持中文的语音
    #            engine.setProperty('voice', voice.id)
    #            break
    #else:
    #    print("没有找到中文语音")

    # 设置语速和音量（可选）
    engine.setProperty('rate', 200)  # 语速
    engine.setProperty('volume', 1)  # 音量（0.0 到 1.0）

    # 将文本转换为语音并播放
    engine.say(text)
    engine.runAndWait()

def text_to_speech(text):
    # 使用 Google TTS 转换中文文本为语音
    tts = gTTS(text, lang='zh')

    # 保存为音频文件
    tts.save("output.mp3")
    
    # 播放音频文件
    playsound.playsound("output.mp3")

while(True):
    # 调用方法
    record_audio()
    result = speech_to_text(output_filename)
    print("识别结果:", result)
    if result == "再见":
        break
    
    #response = get_openai_response(result)
    response = get_openai_response_via_http(result)
    print("OpenAI的回答:", response)
    text_to_speech2(response)
