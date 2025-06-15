import streamlit as st
import requests
import tempfile
import os
from backend import Utils, Generation

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
                    # Save the file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name

                    try:
                        # Process the audio
                        generation = Generation()
                        transcription = generation.transcribe_audio_pytorch(temp_path)
                        summary = generation.summarize_string(transcription)

                        # Display results
                        with st.expander("Transcription", expanded=True):
                            st.text_area("", transcription, height=300)
                        
                        with st.expander("Summary", expanded=True):
                            st.text_area("", summary, height=300)

                        # Clean up
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

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
                        # Download and process YouTube audio
                        temp_path = Utils.download_youtube_audio_to_tempfile(youtube_url)
                        if not temp_path:
                            st.error("Failed to download YouTube audio")
                            return

                        # Process the audio
                        generation = Generation()
                        transcription = generation.transcribe_audio_pytorch(temp_path)
                        summary = generation.summarize_string(transcription)

                        # Display results
                        with st.expander("Transcription", expanded=True):
                            st.text_area("", transcription, height=300)
                        
                        with st.expander("Summary", expanded=True):
                            st.text_area("", summary, height=300)

                        # Clean up
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

if __name__ == "__main__":
    main() 
