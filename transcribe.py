from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

# ==============================
# SETTINGS
# ==============================
VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"
MODEL_SIZE = "base"   # tiny, base, small, medium, large-v2

# ==============================
# STEP 1: Extract Audio
# ==============================
print("Extracting audio from video...")

video = VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(AUDIO_PATH)

print("Audio extracted successfully!")

# ==============================
# STEP 2: Load Whisper Model
# ==============================
print("Loading faster-whisper model...")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

print("Model loaded!")

# ==============================
# STEP 3: Transcribe Audio (English)
# ==============================
print("Transcribing...")

segments_generator, info = model.transcribe(AUDIO_PATH)
segments = list(segments_generator)

print("Detected language:", info.language)

# ==============================
# STEP 4: Translate to Telugu
# ==============================
print("Translating to Telugu...")

translator = GoogleTranslator(source='auto', target='te')

with open("transcript_english.txt", "w", encoding="utf-8") as eng_file, \
     open("transcript_telugu.txt", "w", encoding="utf-8") as tel_file:

    for segment in segments:
        english_text = segment.text.strip()

        # Translate to Telugu
        telugu_text = translator.translate(english_text)

        # Format with timestamps
        eng_line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {english_text}"
        tel_line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {telugu_text}"

        print("\nEN:", eng_line)
        print("TE:", tel_line)

        eng_file.write(eng_line + "\n")
        tel_file.write(tel_line + "\n")

print("\n✅ Transcription + Translation Completed!")
print("Saved as transcript_english.txt and transcript_telugu.txt")
