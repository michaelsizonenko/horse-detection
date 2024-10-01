import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple

IMG_SIZE: Tuple[int, int] = (640, 640)


model = YOLO("best.pt")
def open_video(input_video_path: str, output_video_path: str) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video {input_video_path}")

    fps: int = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc: int = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, IMG_SIZE)

    if not out.isOpened():
        raise IOError(f"Error: Could not open output video {output_video_path}")

    return cap, out

def process_frame(frame: np.ndarray, model: YOLO) -> np.ndarray:
    frame_resized: np.ndarray = cv2.resize(frame, IMG_SIZE)
    frame_rgb: np.ndarray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    results = model.predict(frame_rgb, conf=0.5, classes=0)

    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            height, width = result.orig_img.shape[:2]
            background: np.ndarray = np.zeros((height, width), dtype=np.uint8)

            masks = result.masks.xy
            for mask in masks:
                mask = mask.astype(int)
                cv2.drawContours(background, [mask], -1, 255, thickness=cv2.FILLED)

            frame_bgr: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result_frame: np.ndarray = cv2.bitwise_and(frame_bgr, frame_bgr, mask=background)

            return result_frame

    return frame  # Return the original frame if no masks found
    
def release_resources(cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
    if cap:
        cap.release()
    if out:
        out.release()

def process_video(input_video_path: str, output_video_path: str) -> None:
    cap, out = open_video(input_video_path, output_video_path)

    while cap.isOpened():
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot receive frame.")
            break

        processed_frame: np.ndarray = process_frame(frame, model)

        out.write(processed_frame)

    release_resources(cap, out)
    print(f"Processed video saved to: {output_video_path}")
    return output_video_path






