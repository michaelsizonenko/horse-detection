import cv2
import numpy as np
from ultralytics import YOLO  # Assuming the YOLO model is imported from ultralytics

IMG_SIZE: tuple[int, int] = (640, 640)

class VideoProcessor:
    def __init__(self, input_video_path: str, model_path: str = "best.pt") -> None:
        self.input_video_path: str = input_video_path
        self.model = YOLO(model_path)
        self.cap: cv2.VideoCapture | None = None
        self.out: cv2.VideoWriter | None = None

    def open_video(self,output_video_path:str) -> None:
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open video {self.input_video_path}")

        fps: int = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc: int = cv2.VideoWriter_fourcc(*'vp80') 
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, IMG_SIZE)

        if not self.out.isOpened():
            raise IOError(f"Error: Could not open output video {output_video_path}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_resized: np.ndarray = cv2.resize(frame, IMG_SIZE)
        frame_rgb: np.ndarray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = self.model.predict(frame_rgb, conf=0.5, classes=0,verbose=False)

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                height, width = result.orig_img.shape[:2]
                background: np.ndarray = np.zeros((height, width), dtype=np.uint8)

                masks = result.masks.xy
                for mask in masks:
                    mask = mask.astype(int)
                    cv2.drawContours(background, [mask], -1, 255, thickness=cv2.FILLED)

                # Convert the frame back to BGR for saving
                frame_bgr: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result_frame: np.ndarray = cv2.bitwise_and(frame_bgr, frame_bgr, mask=background)

                return result_frame

        return frame  # Return the original frame if no masks found

    def process_video(self,output_video_path:str) -> str:
        self.open_video(output_video_path=output_video_path)

        while self.cap.isOpened():
            ret: bool
            frame: np.ndarray
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot receive frame.")
                break

            processed_frame: np.ndarray = self.process_frame(frame)

            self.out.write(processed_frame)

        self.release_resources()
        return output_video_path

    def release_resources(self) -> None:
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        print(f"Processed video saved")

