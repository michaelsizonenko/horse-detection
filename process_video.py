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

    def is_moving_right(self, p0, p1, status) -> bool:
        total_motion_x = 0  # Total motion along the X axis for all points
        moving_points_count = 0

        rightward_motion_count = 0
        total_rightward_motion_x = 0
        avg_motion_x = 0

        for i in range(len(p0)):
            if status[i] == 1:  # Process only successfully tracked points
                motion_x = p1[i][0] - p0[i][0]  # Horizontal movement (x)
                total_motion_x += motion_x[0]  # Sum up the motion of all points to estimate global motion
                moving_points_count += 1

        # Estimate global shift (average scene movement)
        if moving_points_count > 0:
            avg_motion_x = total_motion_x / moving_points_count

        # Calculate the relative movement of the object
        for i in range(len(p0)):
            if status[i] == 1:
                motion_x = p1[i][0] - p0[i][0]  # Horizontal movement of the point
                relative_motion_x = motion_x[0] - avg_motion_x  # Relative movement of the point

                if relative_motion_x > 0:  # Rightward movement relative to the scene
                    total_rightward_motion_x += relative_motion_x
                    rightward_motion_count += 1

        # Determine if the object is moving to the right
        if rightward_motion_count > 0:
            avg_rightward_motion_x = total_rightward_motion_x / rightward_motion_count
            return avg_rightward_motion_x > 5  # Threshold for detecting rightward movement

        return False

    def cut_video(self, output_video_path: str,
                          stop_threshold: int = 25) -> str:
        self.open_video(output_video_path=output_video_path)

        # Parameters for Lucas-Kanade Optical Flow
        lk_params = dict(winSize=(21, 21),
                         maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

        # Read the first frame
        ret, old_frame = self.cap.read()
        if not ret:
            print("Error: Unable to read the first frame.")
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Find initial points to track (good features to track)
        p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=50, qualityLevel=0.2, minDistance=70, blockSize=15)

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
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Check if the object is moving to the right
            if self.is_moving_right(p0, p1, status):
                if not horse_moving:
                    print(f"Object started moving right at frame {frame_count}.")
                    horse_moving = True

                self.out.write(frame)
                good_new = p1[status == 1]
                if good_new is not None and len(good_new) > 0:
                    p0 = good_new.reshape(-1, 1, 2)
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
        self.input_video_path = output_video_path
        return output_video_path


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

    def process_video(self, output_video_path: str) -> str:
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

