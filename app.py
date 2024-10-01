import tempfile

import cv2
import streamlit as st

title = st.title("CV Test Task")

video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

if 'cropped_video' not in st.session_state:
    st.session_state.cropped_video = None

if video_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video_capture = cv2.VideoCapture(tfile.name) # convert to opencv compatible format for processing

    st.video(tfile.name)

    if st.button("Crop video"):
        # function to crop video
        st.session_state.cropped_video = tfile.name # cropped video
        st.write("Video cropped successfully.")

    if st.session_state.cropped_video:

        st.video(st.session_state.cropped_video)

        if st.button("Process video"):
            # function to process video (background removal)
            st.session_state.processed_video = tfile.name # processed video
            if st.session_state.cropped_video:
                st.video(st.session_state.processed_video)
                st.write("Background removed successfully.")

            else:
                st.warning("Please crop the video first before processing.")
