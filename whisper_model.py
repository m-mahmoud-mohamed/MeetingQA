import whisper
import pytube as pt
import io
import yt_dlp

class WhisperModel:
    def __init__(self):
        self.model = whisper.load_model("small")

    def download(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio.mp3',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.mp4', '.mp3')
                print(f"Downloaded audio to {audio_file}")
                print(info_dict)
                print(type(info_dict))
                return audio_file
        except yt_dlp.utils.DownloadError as e:
            print(f"An error occurred while downloading the video: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        

    
    def trans(self, audio):
        result = self.model.transcribe(audio)
        print("omar")

        return result["text"]
