# nodes.py

import requests
import numpy as np
import io
import torch
import urllib.parse
import os
import sys
import random
import re
from typing import Dict, Any, List, Tuple
from server import PromptServer
from aiohttp import web
import json
import soundfile as sf
import sounddevice as sd

language_map = {
    "ar": "Arabic", "cs": "Czech", "de": "German", "en": "English",
    "es": "Spanish", "fr": "French", "hi": "Hindi", "hu": "Hungarian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish",
    "zh-cn": "Chinese"
}

DEFAULT_CONFIG = {
    "url": "http://localhost:8020",
    "language": "English",
    "speaker_wav": "default"
}

class Everything(str):
    def __ne__(self, __value: object) -> bool:
        return False

def play_audio_data(audio_data: np.ndarray, sample_rate: int) -> None:
    """Cross-platform audio playback using sounddevice."""
    try:
        # Ensure audio_data is float32 and in range [-1, 1]
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Play the audio
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

class XTTSConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_url": ("STRING", {"default": "http://localhost:8020"}),
                "language": (list(language_map.values()), {"default": language_map["en"]}),
                "speaker_wav": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("TTS_URL", "TTS_LANGUAGE", "TTS_SPEAKER")
    RETURN_NAMES = ("TTS_URL", "TTS_LANGUAGE", "TTS_SPEAKER")
    FUNCTION = "configure_xtts"
    CATEGORY = "Bjornulf"

    def configure_xtts(self, TTS_url, language, speaker_wav):
        return (TTS_url, language, speaker_wav)

    @classmethod
    def IS_CHANGED(cls, TTS_url, language, speaker_wav) -> float:
        return 0.0

class TextToSpeech:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "language": (list(language_map.values()), {"default": "English"}),
                "speaker_wav": ("STRING", {"default": "default"}),
                "autoplay": ("BOOLEAN", {"default": True}),
                "save_audio": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "connect_to_workflow": (Everything("*"), {"forceInput": True}),
                "TTS_URL": ("TTS_URL", {"forceInput": True}),
                "TTS_LANGUAGE": ("TTS_LANGUAGE", {"forceInput": True}),
                "TTS_SPEAKER": ("TTS_SPEAKER", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("AUDIO", "audio_path", "audio_full_path", "audio_duration")
    FUNCTION = "generate_audio"
    CATEGORY = "Bjornulf"
    
    @staticmethod
    def get_language_code(language_name: str) -> str:
        return next((code for code, name in language_map.items() if name == language_name), "en")
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        return re.sub(r'[^\w\s-]', '', text).replace(' ', '_')[:50]

    def create_new_audio(self, text: str, language_code: str, speaker_wav: str, seed: int, TTS_config: Dict) -> io.BytesIO:
        random.seed(seed)
        encoded_text = urllib.parse.quote(text)
        url = f"{TTS_config['url']}/tts_stream?language={language_code}&speaker_wav={speaker_wav}&text={encoded_text}"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            audio_data = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                audio_data.write(chunk)
            
            audio_data.seek(0)
            if audio_data.getbuffer().nbytes == 0:
                raise ValueError("Received empty audio data from server")
            
            return audio_data

        except requests.RequestException as e:
            print(f"Error generating audio: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def process_audio_data(self, autoplay: bool, audio_data: io.BytesIO, save_path: str) -> Tuple[Dict[str, Any], str, float]:
        """
        Process audio data from MP3 BytesIO to numpy array.
        For MP3 support, requires ffmpeg on the system.
        """
        try:
            audio_data.seek(0)
            
            # soundfile can handle MP3 if audioread is installed (which requires ffmpeg)
            # Otherwise, write to temp WAV file and read it
            try:
                audio_np, sample_rate = sf.read(audio_data, format='mp3')
            except Exception:
                # Fallback: write to temporary WAV file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp.write(audio_data.getvalue())
                    tmp_path = tmp.name
                
                try:
                    audio_np, sample_rate = sf.read(tmp_path)
                finally:
                    os.unlink(tmp_path)
            
            # Ensure float32
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32) / 32768.0 if audio_np.dtype == np.int16 else audio_np.astype(np.float32)
            
            num_channels = 1 if audio_np.ndim == 1 else audio_np.shape[1]
            audio_np = audio_np.reshape(-1, num_channels).T if num_channels > 1 else audio_np.reshape(1, -1)
            
            audio_tensor = torch.from_numpy(audio_np)
            
            if autoplay:
                # Play the audio
                play_audio = audio_np.squeeze()
                play_audio_data(play_audio, sample_rate)
            
            duration = len(audio_np[0]) / sample_rate if audio_np.ndim > 1 else len(audio_np) / sample_rate
            
            return ({"waveform": audio_tensor.unsqueeze(0), "sample_rate": sample_rate}, save_path or "", duration)
        
        except Exception as e:
            print(f"Error processing audio data: {e}")
            import traceback
            traceback.print_exc()
            return ({"waveform": torch.zeros(1, 1, 1, dtype=torch.float32), "sample_rate": 22050}, "", 0.0)

    def save_audio_file(self, audio_data: io.BytesIO, save_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(audio_data.getvalue())
            print(f"Audio saved to: {save_path}")
        except Exception as e:
            print(f"Error saving audio file: {e}")

    def load_audio_file(self, file_path: str) -> io.BytesIO:
        try:
            with open(file_path, 'rb') as f:
                audio_data = io.BytesIO(f.read())
            return audio_data
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return io.BytesIO()

    def generate_audio(self, text: str, language: str, speaker_wav: str,
                        autoplay: bool, seed: int, save_audio: bool, overwrite: bool,
                        TTS_URL: str = None, TTS_LANGUAGE: str = None, 
                        TTS_SPEAKER: str = None, connect_to_workflow: Any = None) -> Tuple[Dict[str, Any], str, str, float]:
             
             # Use provided config values or fallback to node parameters
             config = {
                 "url": TTS_URL if TTS_URL is not None else DEFAULT_CONFIG["url"],
                 "language": TTS_LANGUAGE if TTS_LANGUAGE is not None else language,
                 "speaker_wav": TTS_SPEAKER if TTS_SPEAKER is not None else speaker_wav
             }
                 
             language_code = self.get_language_code(config["language"])
             speaker_wav = config["speaker_wav"]
             
             sanitized_text = self.sanitize_text(text)
             save_path = os.path.join("Bjornulf_TTS", config["language"], speaker_wav, f"{sanitized_text}.wav")
             full_path = os.path.abspath(save_path)
             os.makedirs(os.path.dirname(full_path), exist_ok=True)

             if os.path.exists(full_path) and not overwrite:
                 print(f"Using existing audio file: {full_path}")
                 audio_data = self.load_audio_file(full_path)
             else:
                 audio_data = self.create_new_audio(text, language_code, speaker_wav, seed, config)
                 if save_audio:
                     self.save_audio_file(audio_data, full_path)

             audio_output, _, duration = self.process_audio_data(autoplay, audio_data, full_path if save_audio else None)
             return (audio_output, save_path, full_path, duration)

# Scan folder
@PromptServer.instance.routes.post("/bjornulf_TTS_get_voices")
async def get_voices(request):
    try:
        base_path = os.path.join("custom_nodes", "Bjornulf_custom_nodes", "speakers")
        voices_by_language = {}
        
        # Scan each language directory
        for lang_code in language_map.keys():
            lang_path = os.path.join(base_path, lang_code)
            
            if os.path.exists(lang_path):
                # List all .wav files in the language directory
                voice_ids = []
                for file in os.listdir(lang_path):
                    if file.endswith('.wav'):
                        voice_id = os.path.splitext(file)[0]
                        # Include language code in voice ID
                        full_voice_id = f"{lang_code}/{voice_id}"
                        voice_ids.append(full_voice_id)
                
                if voice_ids:  # Only add languages that have voices
                    voices_by_language[lang_code] = {
                        "name": language_map[lang_code],
                        "voices": voice_ids
                    }
        
        if not voices_by_language:
            return web.json_response({"error": "No voice files found"}, status=404)
            
        return web.json_response({"languages": voices_by_language})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return web.json_response({"error": str(e)}, status=500)
