import tempfile
import time
import re
import os
import torch
import torchaudio
import yt_dlp
from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, AutoTokenizer

app = Flask(__name__)

class Utils:
    @staticmethod
    def temporary_file(uploaded_file: bytes) -> str:
        """
        Create a temporary file for the uploaded audio file.
        """
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file)
                temp_file_path = temp_file.name
            return temp_file_path
        
    @staticmethod   
    def clean_transcript(text: str) -> str:
        """
        Clean the transcript text by removing unwanted characters and formatting.
        """
        text = text.replace(",", " ")
        text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        return text.strip()
    
    @staticmethod
    def preprocess_audio(input_path: str) -> str:
        """
        Preprocess the audio file by converting it to mono and resampling to 16000 Hz.
        """
        waveform, sample_rate = torchaudio.load(input_path)
        print(f"📢 Original waveform shape: {waveform.shape}")
        print(f"📢 Original sample rate: {sample_rate}")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print("✅ Converted to mono.")

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            print(f"✅ Resampled to {target_sample_rate} Hz.")
            sample_rate = target_sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            output_path = tmpfile.name

        torchaudio.save(output_path, waveform, sample_rate)
        print(f"✅ Saved preprocessed audio to temporary file: {output_path}")

        return output_path
    
    @staticmethod
    def _format_filename(input_string, chunk_number=0):
        """
        Format the input string to create a valid filename.
        """
        input_string = input_string.strip()
        formatted_string = re.sub(r'[^a-zA-Z0-9\s]', '_', input_string)
        formatted_string = re.sub(r'[\s_]+', '_', formatted_string)
        formatted_string = formatted_string.lower()
        formatted_string += f'_chunk_{chunk_number}'
        return formatted_string

    @staticmethod
    def download_youtube_audio_to_tempfile(youtube_url):
        """
        Download audio from a YouTube video and save it as a WAV file.
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                original_title = info_dict.get('title', 'audio')
                formatted_title = Utils._format_filename(original_title)

            temp_dir = tempfile.mkdtemp()
            output_path_no_ext = os.path.join(temp_dir, formatted_title)

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path_no_ext,
                'quiet': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            expected_output = output_path_no_ext + ".wav"
            timeout = 5
            while not os.path.exists(expected_output) and timeout > 0:
                time.sleep(1)
                timeout -= 1

            if not os.path.exists(expected_output):
                raise FileNotFoundError(f"Audio file was not saved as expected: {expected_output}")

            return expected_output

        except Exception as e:
            print(f"Failed to download {youtube_url}: {e}")
            return None

class Generation:
    def __init__(
            self, 
            summarization_model: str = "vian123/brio-finance-finetuned-v2",
            speech_to_text_model: str = "nyrahealth/CrisperWhisper", 
    ):
        self.summarization_model = summarization_model
        self.speech_to_text_model = speech_to_text_model
        self.device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor_speech = AutoProcessor.from_pretrained(speech_to_text_model)
        self.model_speech = AutoModelForSpeechSeq2Seq.from_pretrained(
            speech_to_text_model,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        ).to(self.device)
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model)

    def transcribe_audio_pytorch(self, file_path: str) -> str:
        """
        Transcribe audio using the PyTorch-based speech-to-text model.
        """
        converted_path = Utils.preprocess_audio(file_path)
        waveform, sample_rate = torchaudio.load(converted_path)
        duration = waveform.shape[1] / sample_rate
        if duration < 1.0:
            print("❌ Audio too short to process.")
            return ""

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_speech,
            tokenizer=self.processor_speech.tokenizer,
            feature_extractor=self.processor_speech.feature_extractor,
            chunk_length_s=5,
            batch_size=1,
            return_timestamps=None,
            torch_dtype=self.dtype,
            device=self.device,
            model_kwargs={"language": "en"},
        )

        try:
            hf_pipeline_output = pipe(converted_path)
            print("✅ HF pipeline output:", hf_pipeline_output)
            return hf_pipeline_output.get("text", "")
        except Exception as e:
            print("❌ Pipeline failed with error:", e)
            return ""

    def summarize_string(self, text: str) -> str:
        """
        Summarize the input text using the summarization model.
        """
        summarizer = pipeline("summarization", model=self.summarization_model, tokenizer=self.summarization_model)
        try:
            if len(text.strip()) < 10:
                return ""

            inputs = self.summarization_tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            truncated_text = self.summarization_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            word_count = len(truncated_text.split())
            min_len = max(int(word_count * 0.5), 30)
            max_len = max(min_len + 20, int(word_count * 0.75))

            summary = summarizer(
                truncated_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error: {e}"

# Flask API endpoints
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not audio_file.filename.endswith('.wav'):
                return jsonify({'error': 'Please upload a WAV file'}), 400

            temp_path = Utils.temporary_file(audio_file.read())
            
            generation = Generation()
            transcription = generation.transcribe_audio_pytorch(temp_path)
            summary = generation.summarize_string(transcription)
            
            os.remove(temp_path)
            
            return jsonify({
                'transcription': transcription,
                'summary': summary
            })

        return jsonify({'error': 'No valid input provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-youtube', methods=['POST'])
def process_youtube():
    try:
        # Get the request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        youtube_url = data.get('youtube_url')
        if not youtube_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400

        print(f"Processing YouTube URL: {youtube_url}")  # Debug log

        # Download YouTube audio
        temp_path = Utils.download_youtube_audio_to_tempfile(youtube_url)
        if not temp_path:
            return jsonify({'error': 'Failed to download YouTube audio'}), 400

        print(f"Downloaded audio to: {temp_path}")  # Debug log

        # Process the audio
        generation = Generation()
        transcription = generation.transcribe_audio_pytorch(temp_path)
        if not transcription:
            return jsonify({'error': 'Failed to transcribe audio'}), 500

        print(f"Transcription completed: {len(transcription)} characters")  # Debug log

        summary = generation.summarize_string(transcription)
        if not summary:
            return jsonify({'error': 'Failed to generate summary'}), 500

        print(f"Summary completed: {len(summary)} characters")  # Debug log

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")  # Debug log

        return jsonify({
            'transcription': transcription,
            'summary': summary
        })

    except Exception as e:
        print(f"Error in process_youtube: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
