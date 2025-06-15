from flask import Flask, render_template, request, jsonify
import os
from backend import Utils, Generation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not audio_file.filename.endswith('.wav'):
                return jsonify({'error': 'Please upload a WAV file'}), 400

            # Save the file temporarily
            temp_path = Utils.temporary_file(audio_file.read())
            
            # Process the audio
            generation = Generation()
            transcription = generation.transcribe_audio_pytorch(temp_path)
            summary = generation.summarize_string(transcription)
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'transcription': transcription,
                'summary': summary
            })

        elif 'youtube_url' in request.form:
            youtube_url = request.form['youtube_url']
            if not youtube_url:
                return jsonify({'error': 'No YouTube URL provided'}), 400

            # Download and process YouTube audio
            temp_path = Utils.download_youtube_audio_to_tempfile(youtube_url)
            if not temp_path:
                return jsonify({'error': 'Failed to download YouTube audio'}), 400

            # Process the audio
            generation = Generation()
            transcription = generation.transcribe_audio_pytorch(temp_path)
            summary = generation.summarize_string(transcription)
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'transcription': transcription,
                'summary': summary
            })

        return jsonify({'error': 'No valid input provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
