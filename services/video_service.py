import tempfile

import streamlit as st

from process_video import VideoProcessor


class VideoService:
    def __init__(self):
        self._initialize_session_state()
        self.videoprocessor = VideoProcessor()

    @staticmethod
    def _initialize_session_state() -> None:
        """Initializes session state variables."""
        session_defaults = {
            "cropped_video": None,
            "processed_video": None,
            "button_pressed": False,
        }
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def upload_and_display_video() -> tempfile.NamedTemporaryFile:
        """Handles video file upload and displays the video."""
        video_file = st.file_uploader("Upload a video", type=["mp4"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.video(tfile.name)
            return tfile
        return None

    def crop_video(self, tfile: tempfile.NamedTemporaryFile) -> None:
        """Crops the video and stores it in session state."""
        if st.button("Crop video"):
            with st.spinner("Cropping video..."):
                cropped_video = self.videoprocessor.cut_video(
                    input_video_path=tfile.name, output_video_path="cropped_video.webm"
                )
                st.session_state["cropped_video"] = cropped_video
                st.success("Video cropped successfully.")

    def process_video(self) -> None:
        """Processes the cropped video and removes the background."""
        if st.session_state["button_pressed"]:
            with st.spinner("Removing background..."):
                processed_video = self.videoprocessor.process_video(
                    input_video_path=st.session_state["cropped_video"],
                    output_video_path="output_video.webm",
                )
                st.session_state["processed_video"] = processed_video
                st.success("Background removed successfully.")
        elif st.button("Remove background"):
            st.session_state["button_pressed"] = True
            st.rerun()

    @staticmethod
    def display_video(key: str) -> None:
        """Displays a video stored in session state."""
        if st.session_state.get(key):
            st.video(st.session_state[key])


video_service = VideoService()
