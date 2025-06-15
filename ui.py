import streamlit as st
import requests
import tempfile
import os
from backend import Utils, Generation

# Backend API URL
BACKEND_URL = "http://151.106.112.219:5000"

def main():
    st.set_page_config(
        page_title="Financial Audio Summarization",
        page_icon="🗣️",
    )

    # Hide Streamlit's default header and footer
    hide_decoration_bar_style = """<style>header {visibility: hidden;}</style>"""
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
    hide_streamlit_footer = """
    <style>#MainMenu {visibility: hidden;}
    footer {visibility: hidden;}</style>
    """
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    st.title("🗣️ Financial Audio Summarization")
    st.markdown("Upload an audio file or provide a YouTube link to transcribe and summarize its content")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Audio File", "YouTube Link"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a WAV audio file",
            type=["wav"],
            help="Upload an audio file for transcription and summarization."
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("Process Audio File", key="process_file"):
                with st.spinner("Processing audio file..."):
                    try:
                        # Prepare the file for API request
                        files = {'audio_file': uploaded_file}
                        
                        # Make API call to backend
                        response = requests.post(f"{BACKEND_URL}/api/process-audio", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Display results
                            with st.expander("Transcription", expanded=True):
                                st.text_area("", data['transcription'], height=300)
                            
                            with st.expander("Summary", expanded=True):
                                st.text_area("", data['summary'], height=300)
                        else:
                            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")

    with tab2:
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the YouTube link of the video you want to summarize."
        )
        if youtube_url:
            st.video(youtube_url)
            if st.button("Process YouTube Video", key="process_youtube"):
                with st.spinner("Processing YouTube video..."):
                    try:
                        # Make API call to backend
                        headers = {'Content-Type': 'application/json'}
                        response = requests.post(
                            f"{BACKEND_URL}/api/process-youtube",
                            json={'youtube_url': youtube_url},
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Display results
                            with st.expander("Transcription", expanded=True):
                                st.text_area("", data['transcription'], height=300)
                            
                            with st.expander("Summary", expanded=True):
                                st.text_area("", data['summary'], height=300)
                        else:
                            error_msg = response.json().get('error', 'Unknown error')
                            st.error(f"Error: {error_msg}")
                            print(f"Backend error: {error_msg}")  # Debug log
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: Could not connect to backend server. Make sure it's running on {BACKEND_URL}")
                        print(f"Request error: {str(e)}")  # Debug log
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")
                        print(f"General error: {str(e)}")  # Debug log

if __name__ == "__main__":
    main() 
