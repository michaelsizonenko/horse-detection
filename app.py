import streamlit as st

from services.video_service import video_service


def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Horse Motion Detection", page_icon=":horse:")
    st.title("Horse Motion Detection Task")

    tfile = video_service.upload_and_display_video()

    if tfile:
        video_service.crop_video(tfile)
        video_service.display_video("cropped_video")

    if st.session_state["cropped_video"]:
        video_service.process_video()
        video_service.display_video("processed_video")


if __name__ == "__main__":
    main()
