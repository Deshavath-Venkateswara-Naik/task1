from moviepy.editor import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment
import os

# ==============================
# SETTINGS
# ==============================
VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"
TELUGU_AUDIO_PATH = "telugu_output.mp3"
FINAL_VIDEO_PATH = "final_dub_video.mp4"
MODEL_SIZE = "base"

# ==============================
# STEP 1: Extract Audio
# ==============================
print("Step 1: Extracting audio from video...")

video = VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(AUDIO_PATH)

print("✅ Audio extracted!")

# ==============================
# STEP 2: English Transcription
# ==============================
print("Step 2: Transcribing to English...")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
segments_gen, info = model.transcribe(AUDIO_PATH)
segments = list(segments_gen)

english_text = ""
for segment in segments:
    english_text += segment.text.strip() + " "

print("✅ English transcription completed!")

# ==============================
# STEP 3: Translate to Telugu
# ==============================
print("Step 3: Translating to Telugu...")

translator = GoogleTranslator(source='auto', target='te')
telugu_text = translator.translate(english_text)

print("✅ Translation completed!")

# ==============================
# STEP 4: Generate Telugu Audio
# ==============================
print("Step 4: Generating Telugu audio...")

MAX_CHARS = 2000
chunks = [telugu_text[i:i+MAX_CHARS] for i in range(0, len(telugu_text), MAX_CHARS)]

temp_files = []

for i, chunk in enumerate(chunks):
    temp_file = f"temp_part_{i}.mp3"
    tts = gTTS(text=chunk, lang="te")
    tts.save(temp_file)
    temp_files.append(temp_file)

# Merge chunks
combined = AudioSegment.empty()

for file in temp_files:
    combined += AudioSegment.from_mp3(file)

combined.export(TELUGU_AUDIO_PATH, format="mp3")

# Remove temporary files
for file in temp_files:
    os.remove(file)

print("✅ Telugu audio generated!")

# ==============================
# STEP 5: Merge Telugu Audio with Video
# ==============================
print("Step 5: Merging Telugu audio with video...")

telugu_audio = AudioFileClip(TELUGU_AUDIO_PATH)

# Match durations safely
if telugu_audio.duration > video.duration:
    telugu_audio = telugu_audio.subclip(0, video.duration)
else:
    video = video.subclip(0, telugu_audio.duration)

# IMPORTANT: MoviePy 1.x uses set_audio()
final_video = video.set_audio(telugu_audio)

final_video.write_videofile(
    FINAL_VIDEO_PATH,
    codec="libx264",
    audio_codec="aac",
    fps=video.fps
)

print("\n🎬 ✅ Final dubbed video saved as:", FINAL_VIDEO_PATH)
print("🔥 Process Completed Successfully!")
