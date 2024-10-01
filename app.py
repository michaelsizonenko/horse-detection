import tempfile

import cv2
import streamlit as st

from cut_video import main
from process_video import process_video

title = st.title("CV Test Task")

video_file = st.file_uploader("Upload a video")

if 'cropped_video' not in st.session_state:
    st.session_state.cropped_video = None

if video_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    st.video(tfile.name)
    print(tfile)

    if st.button("Crop video"):
        cropped_video = main(tfile.name)
        st.session_state.cropped_video = cropped_video # cropped video
        st.write("Video cropped successfully.")

    if st.session_state.cropped_video:

        st.video(st.session_state.cropped_video)
        print(st.session_state.cropped_video)

        if st.button("Process video"):
            # function to process video (background removal)
            processed_video = process_video(st.session_state.cropped_video, f'output_video.webm')
            st.session_state.processed_video = processed_video # processed video
            if st.session_state.cropped_video:
                st.video(st.session_state.processed_video)
                print(st.session_state.processed_video)
                st.write("Background removed successfully.")

            else:
                st.warning("Please crop the video first before processing.")
