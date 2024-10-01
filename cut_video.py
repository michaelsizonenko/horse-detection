import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to check if the horse is moving right
def is_moving_right(p0, p1, status):

    total_motion_x = 0  # Total motion along the X axis for all points
    moving_points_count = 0

    rightward_motion_count = 0
    total_rightward_motion_x = 0

    for i in range(len(p0)):
        if status[i] == 1:  # Process only successfully tracked points
            motion_x = p1[i][0] - p0[i][0]  # Horizontal movement (x)
            total_motion_x += motion_x[0]  # Sum up the motion of all points to estimate global motion
            moving_points_count += 1

    # Estimate global shift (average scene movement)
    if moving_points_count > 0:
        avg_motion_x = total_motion_x / moving_points_count

    # Calculate the relative movement of the horse
    for i in range(len(p0)):
        if status[i] == 1:
            motion_x = p1[i][0] - p0[i][0]  # Horizontal movement of the point
            relative_motion_x = motion_x[0] - avg_motion_x  # Relative movement of the point

            if relative_motion_x > 0:  # Rightward movement relative to the scene
                total_rightward_motion_x += relative_motion_x
                rightward_motion_count += 1

    # Determine if the horse is moving to the right
    if rightward_motion_count > 0:
        avg_rightward_motion_x = total_rightward_motion_x / rightward_motion_count
        return avg_rightward_motion_x > 6  # Threshold for detecting rightward movement

    return False

# Main function
def main(video_path, output_filename="horse_moving_right_fragment.mp4", stop_threshold=15):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define video codec and create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = None

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=1,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Find initial points to track (good features to track)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=50, qualityLevel=0.2, minDistance=20, blockSize=15)

    # Create a mask for drawing the tracks
    mask = np.zeros_like(old_frame)

    horse_moving = False
    start_frame = 0
    frame_count = 0
    stop_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (Lucas-Kanade method)
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Check if the horse is moving to the right
        if is_moving_right(p0, p1, status):
            if not horse_moving:
                print(f"Horse started moving right at frame {frame_count}.")
                horse_moving = True
                start_frame = frame_count

                # Initialize video writer to save the fragment
                out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

            # Draw the tracks
            good_new = p1[status == 1]
            good_old = p0[status == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = int(new[0]), int(new[1])  # Ensure these are integers
                c, d = int(old[0]), int(old[1])
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

            # Write the frame to the output video if horse is moving
            out.write(frame)
            if good_new is not None and len(good_new) > 0:  # Ensure there are good points
                p0 = good_new.reshape(-1, 1, 2)
            stop_counter=0

        # If movement stops, stop writing to the output video
        elif horse_moving:
            stop_counter += 1  # Increment the counter for consecutive non-right movements
            out.write(frame)
            if stop_counter >= stop_threshold:  # Check if it exceeds the threshold
                print(f"Horse stopped moving at frame {frame_count}.")
                horse_moving = False
                if out is not None:
                    out.release()  # Stop saving the fragment
                print(f"Fragment saved as {output_filename}")
                break

        # Display the frame with optical flow
        img = cv2.add(frame, mask)

        # Update old frame and points for the next iteration
        old_gray = frame_gray.copy()
        frame_count += 1

    cap.release()


# Run the main function
if __name__ == "__main__":
    video_path = ''  # Path to the input video file
    main(video_path)
