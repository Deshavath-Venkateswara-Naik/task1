from moviepy.editor import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment
import nltk
import os

# Download tokenizer (first time only)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ==============================
# SETTINGS
# ==============================
VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"
TELUGU_AUDIO_PATH = "telugu_output.mp3"
FINAL_VIDEO_PATH = "final_dubbed_video.mp4"
MODEL_SIZE = "base"

ENGLISH_TRANSCRIPT_FILE = "transcript_english.txt"
TELUGU_TRANSCRIPT_FILE = "transcript_telugu.txt"
TELUGU_SRT_FILE = "telugu_subtitles.srt"

# ==============================
# STEP 1: Extract Audio
# ==============================
print("Step 1: Extracting audio...")
video = VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(AUDIO_PATH)
print("Audio extracted successfully!")

# ==============================
# STEP 2: English Transcription
# ==============================
print("Step 2: Generating English transcript...")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
segments_gen, info = model.transcribe(AUDIO_PATH)
segments = list(segments_gen)

english_full_text = ""
for segment in segments:
    english_full_text += segment.text.strip() + " "

with open(ENGLISH_TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
    f.write(english_full_text)

print("English transcript saved!")

# ==============================
# STEP 3: Translate to Telugu
# ==============================
print("Step 3: Translating to Telugu...")

translator = GoogleTranslator(source='auto', target='te')
telugu_full_text = translator.translate(english_full_text)

with open(TELUGU_TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
    f.write(telugu_full_text)

print("Telugu transcript saved!")

# ==============================
# STEP 4: NLP Segmentation
# ==============================
print("Step 4: Segmenting Telugu using NLTK...")
telugu_sentences = sent_tokenize(telugu_full_text)
print("Total segmented sentences:", len(telugu_sentences))

# ==============================
# STEP 5: Generate SRT
# ==============================
print("Step 5: Creating SRT file...")

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

total_duration = video.duration
num_sentences = len(telugu_sentences)
time_per_sentence = total_duration / max(num_sentences, 1)

with open(TELUGU_SRT_FILE, "w", encoding="utf-8") as srt:
    current_time = 0
    for i, sentence in enumerate(telugu_sentences):
        start_time = current_time
        end_time = current_time + time_per_sentence

        srt.write(f"{i+1}\n")
        srt.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
        srt.write(sentence.strip() + "\n\n")

        current_time = end_time

print("SRT file created!")

# ==============================
# STEP 6: Generate Telugu Audio
# ==============================
print("Step 6: Generating Telugu audio...")

MAX_CHARS = 2000
chunks = [telugu_full_text[i:i+MAX_CHARS] for i in range(0, len(telugu_full_text), MAX_CHARS)]

temp_files = []

for i, chunk in enumerate(chunks):
    temp_file = f"temp_part_{i}.mp3"
    tts = gTTS(text=chunk, lang='te')
    tts.save(temp_file)
    temp_files.append(temp_file)

# Merge audio chunks
combined = AudioSegment.empty()
for file in temp_files:
    combined += AudioSegment.from_mp3(file)

combined.export(TELUGU_AUDIO_PATH, format="mp3")

# Remove temp files
for file in temp_files:
    os.remove(file)

print("Telugu audio generated!")

# ==============================
# STEP 7: Merge Telugu Audio with Video
# ==============================
print("Step 7: Merging Telugu audio with video...")

telugu_audio = AudioFileClip(TELUGU_AUDIO_PATH)

# Adjust duration safely
if telugu_audio.duration > video.duration:
    telugu_audio = telugu_audio.subclip(0, video.duration)
else:
    video = video.subclip(0, telugu_audio.duration)

final_video = video.set_audio(telugu_audio)

final_video.write_videofile(
    FINAL_VIDEO_PATH,
    codec="libx264",
    audio_codec="aac"
)

print("\n🎬 ✅ Final dubbed video saved as:", FINAL_VIDEO_PATH)
print("Process completed successfully!")
