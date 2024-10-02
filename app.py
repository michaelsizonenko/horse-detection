import tempfile

import streamlit as st

from process_video import VideoProcessor


def main() -> None:

    st.title("CV Test Task")

    video_file = st.file_uploader("Upload a video")

    if "cropped_video" not in st.session_state:
        st.session_state["cropped_video"] = None
    if "processed_video" not in st.session_state:
        st.session_state["processed_video"] = None

    if video_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        videoprocessor = VideoProcessor(tfile.name)

        if st.button("Crop video"):
            with st.spinner("Cropping video..."):
                cropped_video = videoprocessor.cut_video("horse_moving_right_fragment.webm")
                st.session_state["cropped_video"] = cropped_video
                st.write("Video cropped successfully.")

        if st.session_state["cropped_video"] is not None:
            st.video(st.session_state["cropped_video"])

            if st.button("Process video"):
                with st.spinner("Processing video... (Removing background)"):
                    processed_video = videoprocessor.process_video("output_video.webm")
                    st.session_state["processed_video"] = processed_video
                    st.write("Background removed successfully.")

        if st.session_state["processed_video"] is not None:
            st.video(st.session_state["processed_video"])


if __name__ == "__main__":
    main()
