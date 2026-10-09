"""Optional voice control (pip install SpeechRecognition PyAudio).

Runs in its own thread and pushes recognised phrases into a callback (a queue.put).
Uses Google's free web recogniser, so it needs an internet connection.
"""
from __future__ import annotations

import threading
import time


class VoiceControl(threading.Thread):
    def __init__(self, language: str, callback):
        super().__init__(daemon=True, name="voice")
        self.language = language
        self.callback = callback
        self.error = ""
        self.listening = False
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self.error = "pip install SpeechRecognition PyAudio"
            return
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
        except Exception as exc:
            self.error = f"microphone: {exc}"
            return
        self.listening = True
        while not self._stop_evt.is_set():
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                text = recognizer.recognize_google(audio, language=self.language)
                self.callback(text.lower())
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                self.error = f"speech service: {exc}"
                time.sleep(2)
            except Exception:
                time.sleep(1)
