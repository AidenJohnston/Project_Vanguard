import os
import yt_dlp
import time
import psutil
from moviepy import VideoFileClip
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm

def download_video(url, save_path):
    print(f"Downloading video from {url}...")
    ydl_opts = {
        'outtmpl': f'{save_path}\\%(title)s.%(ext)s',
        'format': 'bestvideo+bestaudio/best',
        #this was on my personal pc, ignore all the filepaths I know I'm a bad programmer, this script is just for preprocessing
        'ffmpeg_location': r"C:\Users\aiden\AppData\Local\Programs\Python\Python313\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete!")

def process_audio_with_moviepy(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm(os.listdir(input_dir), desc="Processing Files", unit="file"):
        if file_name.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv")):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
            try:
                video = VideoFileClip(input_path)
                audio = video.audio
                audio.write_audiofile(output_path, fps=16000, codec="pcm_s16le")
                print(f"Processed: {file_name} -> {output_path}")
                audio.close()
                video.close()
                os.remove(input_path)
                print(f"Deleted: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {type(e).__name__}: {e}")

def split_audio_with_pydub_and_delete(input_dir, output_dir, chunk_duration=30, start_show=1):
    os.makedirs(output_dir, exist_ok=True)
    show_counter = start_show
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_dir, file_name)
            show_folder_name = f"Show{show_counter}"
            file_output_dir = os.path.join(output_dir, show_folder_name)
            os.makedirs(file_output_dir, exist_ok=True)

            try:
                audio = AudioSegment.from_file(input_path)
                total_duration = len(audio) // 1000

                for start_time in range(0, total_duration, chunk_duration):
                    end_time = min(start_time + chunk_duration, total_duration)
                    chunk = audio[start_time * 1000:end_time * 1000]
                    chunk_name = f"{show_folder_name}_Chunk{start_time // chunk_duration + 1}.wav"
                    chunk_path = os.path.join(file_output_dir, chunk_name)
                    chunk.export(chunk_path, format="wav")
                    print(f"Saved: {chunk_path}")

                audio = None
                time.sleep(0.5)

                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    if input_path in str(proc.info['open_files']):
                        print(f"Process {proc.info['name']} (PID: {proc.info['pid']}) is holding the file.")

                os.remove(input_path)
                print(f"Deleted: {file_name}")
                show_counter += 1

            except Exception as e:
                print(f"Error processing {file_name}: {type(e).__name__}: {e}")

def extract_audio_features(file_path, sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        note_density = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        pitch_class = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr))
        hnr = np.mean(librosa.effects.harmonic(y))
        pitch_variation = np.std(librosa.yin(y=y, sr=sr, fmin=30, fmax=2000))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = [np.mean(m) for m in mfccs]
        rms_energy = np.mean(librosa.feature.rms(y=y))
        dynamic_range = np.ptp(librosa.feature.rms(y=y))
        crescendo_detection = np.max(np.diff(librosa.feature.rms(y=y)))
        transient_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        silence_ratio = np.sum(y == 0) / len(y)

        feature_dict = {
            "Tempo": tempo,
            "Beat Strength": beat_strength,
            "Zero Crossing Rate": zcr,
            "Note Density": note_density,
            "Chroma Features": chroma,
            "Pitch Class Distribution": pitch_class,
            "Harmonic-to-Noise Ratio": hnr,
            "Pitch Variation": pitch_variation,
            "Spectral Centroid": spectral_centroid,
            "Spectral Bandwidth": spectral_bandwidth,
            "Spectral Contrast": spectral_contrast,
            "RMS Energy": rms_energy,
            "Dynamic Range": dynamic_range,
            "Crescendo Detection": crescendo_detection,
            "Transient Strength": transient_strength,
            "Silence Ratio": silence_ratio,
        }

        for i, mfcc_val in enumerate(mfccs_mean):
            feature_dict[f"MFCC_{i+1}"] = mfcc_val

        return feature_dict

    except Exception as e:
        print(f"Error processing {file_path}: {type(e).__name__}: {e}")
        return None

def process_all_shows(chunks_dir, output_dir):
    for show_folder in os.listdir(chunks_dir):
        show_path = os.path.join(chunks_dir, show_folder)

        if os.path.isdir(show_path):
            print(f"Processing show: {show_folder}")
            show_output_dir = os.path.join(output_dir, show_folder)
            os.makedirs(show_output_dir, exist_ok=True)

            for chunk_file in os.listdir(show_path):
                if chunk_file.endswith(".wav"):
                    chunk_path = os.path.join(show_path, chunk_file)
                    print(f"Extracting features from: {chunk_file}")
                    features = extract_audio_features(chunk_path)
                    if features:
                        features["File Name"] = chunk_file
                        chunk_csv_path = os.path.join(show_output_dir, f"{os.path.splitext(chunk_file)[0]}.csv")
                        pd.DataFrame([features]).to_csv(chunk_csv_path, index=False)
                        print(f"Saved CSV: {chunk_csv_path}")
                        os.remove(chunk_path)
                        print(f"Deleted .wav file: {chunk_path}")

            if not os.listdir(show_path):
                try:
                    os.rmdir(show_path)
                    print(f"Deleted empty folder: {show_path}")
                except Exception as e:
                    print(f"Error deleting folder {show_path}: {type(e).__name__}: {e}")

def main():
    links_input = input("Paste YouTube links (separate by commas or newlines): ")
    links = [link.strip() for link in links_input.replace(",", "\n").split("\n")]

    show_number = int(input("Enter the show number to start with (e.g., Show1): "))

    #you can ignore ts, this was on my personal pc and this script is just for preprocessing
    save_path = r"C:\Users\aiden\Desktop\Project Vanguard\Raw Videos"
    audio_output_dir = r"C:\Users\aiden\Desktop\Project Vanguard\To .WAV"
    chunk_output_dir = r"C:\Users\aiden\Desktop\Project Vanguard\Chunks"
    feature_output_dir = r"C:\Users\aiden\Desktop\Project Vanguard\Input_Data"

    for link in links:
        download_video(link, save_path)
        process_audio_with_moviepy(save_path, audio_output_dir)
        split_audio_with_pydub_and_delete(audio_output_dir, chunk_output_dir, start_show=show_number)
        show_number += 1

    process_all_shows(chunk_output_dir, feature_output_dir)

    print("All processes complete")

if __name__ == "__main__":
    main()