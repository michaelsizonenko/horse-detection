import cv2
import numpy as np
from ultralytics import YOLO  # Assuming the YOLO model is imported from ultralytics


class VideoProcessor:
    def __init__(self, model_path: str = "best.pt") -> None:
        self.model = YOLO(model_path)
        self.video_size: tuple[int, int] | None = None
        self.cap: cv2.VideoCapture | None = None
        self.out: cv2.VideoWriter | None = None

    def open_video(self, input_video_path: str, output_video_path: str) -> None:
        """Opens the input video for reading and sets up the output video writer for saving processed frames."""
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open video {input_video_path}")

        fps: int = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc: int = cv2.VideoWriter_fourcc(*'vp80')
        if self.video_size is None:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_size = (frame_width, frame_height)

        self.out = cv2.VideoWriter(
            output_video_path, fourcc, fps, self.video_size)

        if not self.out.isOpened():
            raise IOError(
                f"Error: Could not open output video {output_video_path}")

    @staticmethod
    def calculate_total_motion(p0, p1, status) -> tuple[float, int]:
        """Calculates the total motion in the x-direction for tracked point"""
        total_motion_x = 0
        moving_points_count = 0
        for i in range(len(p0)):
            if status[i] == 1:
                motion_x = p1[i][0] - p0[i][0]
                total_motion_x += motion_x[0]
                moving_points_count += 1
        return total_motion_x, moving_points_count

    @staticmethod
    def calculate_relative_motion(p0, p1, status, avg_motion_x) -> tuple[float, int]:
        """Calculates the relative motion compared to the average motion"""
        total_rightward_motion_x = 0
        rightward_motion_count = 0
        for i in range(len(p0)):
            if status[i] == 1:
                motion_x = p1[i][0] - p0[i][0]
                relative_motion_x = motion_x[0] - avg_motion_x
                if relative_motion_x > 0:
                    total_rightward_motion_x += relative_motion_x
                    rightward_motion_count += 1
        return total_rightward_motion_x, rightward_motion_count

    def is_moving_right(self, p0, p1, status) -> bool:
        """Determines whether the tracked object is moving to the right based on the average motion of the points."""
        total_motion_x, moving_points_count = self.calculate_total_motion(p0, p1, status)

        if moving_points_count == 0:
            return False

        avg_motion_x = total_motion_x / moving_points_count
        total_rightward_motion_x, rightward_motion_count = self.calculate_relative_motion(p0, p1, status, avg_motion_x)
        if rightward_motion_count > 0:
            avg_rightward_motion_x = total_rightward_motion_x / rightward_motion_count
            return avg_rightward_motion_x > 5  # Threshold for detecting rightward movement

        return False

    @staticmethod
    def read_first_frame(cap) -> tuple[np.ndarray, np.ndarray]:
        """Reads and returns the first frame of the video along with its grayscale version."""
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Error: Unable to read the first frame.")
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def update_tracking_points(p1, status, p0) -> np.ndarray:
        """Updates the points for tracking based on their status and returns the updated points."""
        good_new = p1[status == 1]
        if good_new is not None and len(good_new) > 0:
            return good_new.reshape(-1, 1, 2)
        return p0

    def cut_video(self, input_video_path: str, output_video_path: str,
                  stop_threshold: int = 25) -> str | None:
        """Cuts the video until the object stops moving to the right based on optical flow"""
        self.open_video(input_video_path, output_video_path=output_video_path)

        # Parameters for Lucas-Kanade Optical Flow
        lk_params = dict(winSize=(21, 21),
                         maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

        old_frame, old_gray = self.read_first_frame(self.cap)

        # Find initial points to track (good features to track)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=50, qualityLevel=0.2, minDistance=70, blockSize=15)

        horse_moving = False
        frame_count = 0
        stop_counter = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot receive frame.")
                break

            # Calculate optical flow (Lucas-Kanade method)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)

            # Check if the object is moving to the right
            if self.is_moving_right(p0, p1, status):
                if not horse_moving:
                    print(
                        f"Object started moving right at frame {frame_count}.")
                    horse_moving = True

                self.out.write(frame)
                p0 = self.update_tracking_points(p1, status, p0)
                stop_counter = 0

            elif horse_moving:
                stop_counter += 1
                self.out.write(frame)
                if stop_counter >= stop_threshold:
                    print(f"Object stopped moving at frame {frame_count}.")
                    horse_moving = False
                    break

            old_gray = frame_gray.copy()
            frame_count += 1

        self.release_resources()
        return output_video_path

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes frame and removes the background."""
        frame_resized: np.ndarray = cv2.resize(
            frame, (640, 640))  # resize frame to input to yolo
        frame_rgb: np.ndarray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            frame_rgb, conf=0.5, classes=0, verbose=False)

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                height, width = result.orig_img.shape[:2]
                background: np.ndarray = np.zeros(
                    (height, width), dtype=np.uint8)

                masks = result.masks.xy
                for mask in masks:
                    mask = mask.astype(int)
                    cv2.drawContours(
                        background, [mask], -1, 255, thickness=cv2.FILLED)

                # Convert the frame back to BGR for saving
                frame_bgr: np.ndarray = cv2.cvtColor(
                    frame_rgb, cv2.COLOR_RGB2BGR)
                result_frame: np.ndarray = cv2.bitwise_and(
                    frame_bgr, frame_bgr, mask=background)

                return result_frame

        return frame  # Return the original frame if no masks found

    def process_video(self, input_video_path: str, output_video_path: str) -> str:
        """Processes every frame of video and removes the background."""
        self.video_size = (640, 640)
        self.open_video(input_video_path, output_video_path=output_video_path)

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
        """Releases the resources used for video capture and video writing."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        print(f"Processed video saved")
