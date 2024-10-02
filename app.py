import tempfile

import streamlit as st

from process_video import VideoProcessor
from cut_video import main as cut_video_main


class VideoService:
    def __init__(self):
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initializes session state variables."""
        session_defaults = {
            "cropped_video": None,
            "processed_video": None,
            "button_pressed": False,
        }
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def upload_and_display_video(self) -> tempfile.NamedTemporaryFile:
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
                cropped_video = cut_video_main(tfile.name)
                st.session_state["cropped_video"] = cropped_video
                st.success("Video cropped successfully.")

    def process_video(self) -> None:
        """Processes the cropped video and removes the background."""
        if st.session_state["button_pressed"]:
            with st.spinner("Removing background..."):
                processor = VideoProcessor(st.session_state["cropped_video"])
                processed_video = processor.process_video("output_video.webm")
                st.session_state["processed_video"] = processed_video
                st.success("Background removed successfully.")
        elif st.button("Remove background"):
            st.session_state["button_pressed"] = True
            st.rerun()

    def display_video(self, key: str) -> None:
        """Displays a video stored in session state."""
        if st.session_state.get(key):
            st.video(st.session_state[key])


def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Horse Motion Detection Task", page_icon=":horse:")
    st.title("Horse Motion Detection Task")

    video_service = VideoService()

    tfile = video_service.upload_and_display_video()

    if tfile:
        video_service.crop_video(tfile)
        video_service.display_video("cropped_video")

    if st.session_state["cropped_video"]:
        video_service.process_video()
        video_service.display_video("processed_video")


if __name__ == "__main__":
    main()
