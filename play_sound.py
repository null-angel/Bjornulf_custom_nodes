import os
import io
import sys
import torch
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import sounddevice as sd

class Everything(str):
    def __ne__(self, __value: object) -> bool:
        return False

def play_audio_data(audio_data: np.ndarray, sample_rate: int) -> None:
    """
    Platform-independent audio playback using sounddevice.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate in Hz
    """
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
        sd.wait()  # Wait for playback to finish
    except Exception as e:
        print(f"Error playing audio: {e}")
        import traceback
        traceback.print_exc()

def load_audio_file(audio_path: str) -> tuple:
    """
    Load audio file and return (audio_data, sample_rate).
    Supports WAV, FLAC, OGG, and other formats supported by soundfile.
    """
    try:
        audio_data, sample_rate = sf.read(audio_path)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        raise

class PlayAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (Everything("*"), {"forceInput": True}),
            },
            "optional": {
                "AUDIO": ("AUDIO", {"forceInput": True}),
                "audio_path": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = (Everything("*"),)
    RETURN_NAMES = ("anything",)
    FUNCTION = "execute"
    CATEGORY = "audio"
    
    def play_audio(self, anything, AUDIO=None, audio_path=None):
        """Play audio from AUDIO dict, file path, or default bell sound."""
        try:
            audio_data = None
            sample_rate = None
            
            # Case 1: AUDIO input is provided
            if AUDIO is not None:
                if isinstance(AUDIO, dict) and 'waveform' in AUDIO:
                    waveform = AUDIO['waveform']
                    sample_rate = AUDIO.get('sample_rate', 44100)
                    
                    if isinstance(waveform, torch.Tensor):
                        audio_data = waveform.cpu().numpy()
                    else:
                        audio_data = np.asarray(waveform)
                    
                    # Squeeze out batch/channel dimensions if needed
                    if audio_data.ndim > 1:
                        # If it's (1, 1, samples) or (1, samples), squeeze it
                        audio_data = audio_data.squeeze()
                else:
                    raise ValueError(f"Unsupported AUDIO type: {type(AUDIO)}")
            
            # Case 2: audio_path is provided
            elif audio_path and os.path.exists(audio_path):
                audio_data, sample_rate = load_audio_file(audio_path)
            
            # Case 3: Default to bell sound
            else:
                audio_file = os.path.join(os.path.dirname(__file__), 'bell.m4a')
                if not os.path.exists(audio_file):
                    raise FileNotFoundError(f"Default bell.m4a not found at {audio_file}")
                audio_data, sample_rate = load_audio_file(audio_file)
            
            # Play the sound
            if audio_data is not None and sample_rate is not None:
                play_audio_data(audio_data, sample_rate)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
            import traceback
            traceback.print_exc()

    def execute(self, anything, AUDIO=None, audio_path=None):
        self.play_audio(anything, AUDIO, audio_path)
        return (anything,)
    
    @classmethod
    def IS_CHANGED(cls, anything, AUDIO=None, audio_path=None, *args):
        return float("NaN")
