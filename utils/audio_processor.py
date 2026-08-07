import yt_dlp
from pydub import AudioSegment
import os
from rich import print

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio( url : str) -> str:
    '''This function will get an URL of a youtube vedio and will donload it.'''
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        'format' : 'bestaudio/best',
        'outtmpl' : output_path,
        'postprocessors' : [
            {
                "key" : "FFmpegExtractAudio",
                "preferredcodec" : "wav",
                "preferredquality" : "192"
            }
        ],
        'quiet' : True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        final_path = ydl.prepare_filename(info)
        base, _ = final_path.rsplit(".",1)
        final_path = f"{base}.wav"
    return final_path

def convert_to_wav_mono(file_path : str) -> str:
    ''' It will convert an audio file to WAV format with mono channel and 16kHz frame rate'''
    # Load Audio
    audio = AudioSegment.from_file(file = file_path)

    #Set to mono channel and 16kHz frame rate
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Setting output path
    base, _ = os.path.splitext(file_path)
    output_path = f"{base}_16k_mono.wav"

    #Export as WAV
    audio.export(output_path, format="wav")

    return output_path

def audio_to_chunk(file_path : str, chunk_minutes : int = 10) -> list:
    '''This will split an audio file into fixed length chunks(by default and fixed at 10 minutes)'''
    audio = AudioSegment.from_wav(file = file_path)

    # Chunk length in milliseconds
    chunk_length_ms = chunk_minutes*60*1000

    total_length = len(audio)

    chunk_paths = []

    for i, start in enumerate(range(0, total_length, chunk_length_ms)):
        chunk = audio[start : start+chunk_length_ms]
        chunk_path = f'{file_path}_chunk_{i}.wav'
        chunk.export(chunk_path, format='wav')
        chunk_paths.append(chunk_path)

    return chunk_paths

def process_audio_from_url(path : str) -> list:
    if path.startswith("https://") or path.startswith("http://"):
        print("Detected a Youtube URL... ")
        print("Downloading audio....")
        file_path = download_youtube_audio(url = path)
    else:
        print("Detected a local file...")
        print("Converting to WAV format...")
        file_path = convert_to_wav_mono(file_path = path)

    chunk_paths = audio_to_chunk(file_path = file_path)

    return chunk_paths





 