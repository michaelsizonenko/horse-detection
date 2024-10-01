import torch
from ultralytics import YOLO
import cv2
import numpy as np

IMG_SIZE = (640, 640)
class VideoProcessor:
    def __init__(self, input_video_path, output_video_path, model_path="best.pt"):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model = YOLO(model_path)
        self.cap = None
        self.out = None

    def open_video(self):
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open video {self.input_video_path}")


        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, fps, IMG_SIZE)

        if not self.out.isOpened():
            raise IOError(f"Error: Could not open output video {self.output_video_path}")

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = self.model.predict(frame_rgb, conf=0.5, classes=0)

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                height, width = result.orig_img.shape[:2]
                background = np.zeros((height, width), dtype=np.uint8)

                masks = result.masks.xy
                for mask in masks:
                    mask = mask.astype(int)
                    cv2.drawContours(background, [mask], -1, 255, thickness=cv2.FILLED)

                # Convert the frame back to BGR for saving
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result_frame = cv2.bitwise_and(frame_bgr, frame_bgr, mask=background)

                return result_frame

        return frame  # Return the original frame if no masks found

    def process_video(self):
        self.open_video()  

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot receive frame.")
                break

            processed_frame = self.process_frame(frame)

            self.out.write(processed_frame)


        self.release_resources()

    def release_resources(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {self.output_video_path}")


# Usage
input_video = "Horse.mp4"  # Path to input video
output_video = "output_video.mp4"  # Path to save the processed video

try:
    processor = VideoProcessor(input_video, output_video)
    processor.process_video()
except Exception as e:
    print(f"An error occurred: {e}")
