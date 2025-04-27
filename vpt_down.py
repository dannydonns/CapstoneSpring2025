import cv2
import mediapipe as mp
import numpy as np
from pose_lib import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from mss import mss

with mss() as sct:
    monitor = sct.monitors[2]


class_name='down_R'
pose_samples_folder = 'pose_csv'
threshold_begin = 4
threshold_end = 2
# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=20,
    top_n_by_mean_distance=5)

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=5,
    alpha=0.1)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=threshold_begin,
    exit_threshold=threshold_end)

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=500,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=7)

# Run classification on a video.
import os
import tqdm

from mediapipe.python.solutions import drawing_utils as mp_drawing


video_path = 0
frame_idx = 0
output_frame = None
# Initialize tracker.
pose_tracker = mp_pose.Pose()
video_cap = cv2.VideoCapture(video_path)

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
while True:
    # Get next frame of the video.
    success, input_frame = video_cap.read()

    if not success:
        break

    # Run pose tracker.
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    # input_frame = cv2.resize(input_frame,(video_height, video_height))
    result = pose_tracker.process(image=input_frame)
    pose_landmarks = result.pose_landmarks

    # Draw pose prediction.
    output_frame = input_frame.copy()
    if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

    if pose_landmarks is not None:
        # Get landmarks.
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                   for lmk in pose_landmarks.landmark], dtype=np.float32)
        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks)

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(pose_classification)

        # Count repetitions.
        repetitions_count = repetition_counter(pose_classification_filtered)
    else:
        # No pose => no classification on current frame.
        pose_classification = None

        # Still add empty classification to the filter to maintaing correct
        # smoothing for future frames.
        pose_classification_filtered = pose_classification_filter(dict())
        pose_classification_filtered = None

        # Don't update the counter presuming that person is 'frozen'. Just
        # take the latest repetitions count.
        repetitions_count = repetition_counter.n_repeats

    # Draw classification plot and repetition counter.
    output_frame = pose_classification_visualizer(
        frame=output_frame,
        pose_classification=pose_classification,
        pose_classification_filtered=pose_classification_filtered,
        repetitions_count=repetitions_count)

    output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)

    try:
        cv2.rectangle(output_frame, (int(20 / 640 * frame_width), int(420 / 480 * frame_height)),
                      (int(270 / 640 * frame_width), int(450 / 480 * frame_height)), (255, 255, 255), 3)

        conf = pose_classification[class_name]
        bar_right = np.interp(conf, (0, threshold_begin), (int(270 / 640 * frame_width), int(20 / 640 * frame_width)))
        if conf > threshold_begin:
            output_frame[int(420 / 480 * frame_height):int(450 / 480 * frame_height), int(20 / 640 * frame_width):int(bar_right),
            :] = [0, 128, 0]
        if conf <1:
            output_frame[int(420 / 480 * frame_height):int(450 / 480 * frame_height), int(20 / 640 * frame_width): int(bar_right),
            :] = [0, 0, 128]
        if 1 <= conf <= threshold_begin:
            output_frame[int(420 / 480 * frame_height):int(450 / 480 * frame_height), int(20 / 640 * frame_width): int(bar_right),
            :] = [128, 0, 0]
    except:
        pass
    # # Save the output frame.
    # out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
    #
    # Show intermediate frames of the video to track progress.
    # if frame_idx % 30 == 0:
    #     show_image(output_frame)
    cv2.imshow('VPT action', output_frame)
    #
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    print(repetitions_count)

    frame_idx += 1
