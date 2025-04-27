import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from mss import mss

with mss() as sct:
    monitor = sct.monitors[2]

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_3D(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    p1 = a - b
    p2 = c - b
    cosangle = p1.dot(p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    angle = np.abs(np.arccos(cosangle) * 180.0 / np.pi)

    # x2 = c[0] - b[0]
    # y2 = c[1] - b[1]
    # radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # angle = np.abs(radians * 180.0 / np.pi)
    print(angle)
    if angle > 180.0:
        angle = 360- angle

    return angle


cap = cv2.VideoCapture(0)
counter = 0
status = None
counter_right = 0
status_right = None
with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # get image stream and process
    while cap.isOpened():
    # while 1:
        ret, frame = cap.read()

        # web camera
        frame = cv2.flip(frame, 2)

        # monitor 3
        # frame = np.array(sct.grab(monitor))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ############################ left side
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            percentage = np.interp(angle, (20, 180), (100, 0))
            bar = np.interp(angle, (20, 180), (int(350/640 * frame_width), int(620/640*frame_width)))

            # Visualize angle
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 150:
                status = "down"
            if angle < 30 and status == 'down':
                status = "up"
                counter += 1
                # print(counter)

            # Motion data
            cv2.putText(image, str(counter),
                        (int(560/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            # # Status data
            # cv2.putText(image, status,
            #             (60, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, f'{int(percentage)} %',
                        (int(500/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(image, (int(620/640*frame_width), int(420/480*frame_height)),
                          (int(350/640*frame_width), int(450/480*frame_height)), (255, 255, 255), 3)
            if angle < 30:
                image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 128, 0]
            if angle > 150:
                image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 0, 128]
            if 30 <= angle <= 150:
                image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [128, 0, 0]

            #################### right side
            # Get coordinates
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              # landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                              ]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
                           ]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
                           ]

            # Calculate angle
            angle_right = calculate_angle_3D(shoulder_right, elbow_right, wrist_right)
            percentage_right = np.interp(angle_right, (20, 180), (100, 0))
            bar_right = np.interp(angle_right, (20, 180), (int(270/640 * frame_width), int(20/640*frame_width)))

            # Visualize angle
            cv2.putText(image, str(int(angle_right)),
                        tuple((np.multiply(elbow_right[0:2], [frame_width, frame_height])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle_right > 160:
                status_right = "down"
            if angle_right < 30 and status_right == 'down':
                status_right = "up"
                counter_right += 1
                # print(counter)

            # Motion data
            cv2.putText(image, str(counter_right),
                        (int(10/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            cv2.putText(image, f'{int(percentage_right)} %',
                        (int(150/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(image, (int(20 / 640 * frame_width), int(420 / 480 * frame_height)),
                          (int(270 / 640 * frame_width), int(450 / 480 * frame_height)), (255, 255, 255), 3)
            if angle_right < 30:
                image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width):int(bar_right), :] = [0, 128, 0]
            if angle_right > 160:
                image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [0, 0, 128]
            if 30 <= angle_right <= 160:
                image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [128, 0, 0]

        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('VPT action', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
