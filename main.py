########################################################################
# BrilliantAI Code
# Splash Screen based on spinndesign.com

########################################################################
## IMPORTS
########################################################################
# from yolo_model import *

from pose_lib import *
import threading
from kivy.lang import Builder
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
import time
from datetime import datetime
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivymd.uix.label import MDLabel
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, IRightBodyTouch
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDRoundFlatIconButton, MDFloatingActionButton
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.slider import MDSlider
from kivymd.uix.swiper import MDSwiper
from kivymd.uix.carousel import MDCarousel
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from frame_processing import *
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# cap = cv2.VideoCapture(0)

# ret, img = cap.read()

# cap.release()

#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation


#from android.permissions import request_permissions, Permission
#request_permissions([Permission.CAMERA])

import os
dirname = os.path.dirname(__file__)

import numpy as np
import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import threading
import copy

Window.fullscreen = False

current_window = 'none'


# Globals

class_name='down_R'
pose_samples_folder = 'assets'
threshold_begin = 8.2
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

#######
#Color Method

def calculate_bgr(percent):

    if percent < 50:
        return (0,percent*5,255)
    else:
        return (0,255,255-((percent-50)*5))

def calculate_bgr_half(percent):

    percent = (50 - abs(percent-50))*2
    #print(percent)

    if percent < 50:
        #print(0,percent*5,255)
        return (0,percent*5,255)
    else:
        #print(0,255,255-((percent-50)*5))
        return (0,255,255-((percent-50)*5))

    


########################################################################
## Calculation Definitions
########################################################################

class PoseAnalysis:
    def __init__(self):
        self.pose_data = []
        self.feature_data = []
        self.feature_rw_data = []
        self.feature_n_data = []

        self.key_pose_data = []
        self.key_feature_data = []


    def add_data(self,landmarks, features):
        self.pose_data.append(landmarks)
        self.feature_data.append(features)
        

def calculate_rotation_matrix_x(angle):

    return [[1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]]

def calculate_rotation_matrix_y(angle):

    return [[np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]]

def calculate_new_landmarks(landmarks):

    #for l in landmarks:
    #    l.x = l.x*200
    #    l.y = l.y*200
    #    l.z = l.z*200

    hip_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    hip_left = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])

    hip_left_new = hip_left - hip_right

    #print(hip_left_new)

    x_angle = np.arctan(hip_left_new[1]/hip_left_new[2])
    x_rot_mat = calculate_rotation_matrix_x(x_angle)

    hip_left_new = np.matmul(x_rot_mat, hip_left_new)

   

    y_angle = np.arctan(hip_left_new[0]/hip_left_new[2])
    y_rot_mat = calculate_rotation_matrix_y(-y_angle)

    #print(hip_left_new)
 
    rot_mat = np.matmul(x_rot_mat, y_rot_mat)

    hip_left_new = np.matmul(y_rot_mat, hip_left_new)

    #print(hip_left_new)

    #print(hip_left_new)
 
    new_landmarks = list(landmarks)

    count = 0
    for l in new_landmarks:
        #print(l)
        l = np.array([l.x, l.y, l.z]) - hip_right
        #print(l)
        #print(hip_right)
        #print(rot_mat)
        l = np.matmul(rot_mat, l)
        #print(l)
        new_landmarks[count].x = l[0]
        new_landmarks[count].y = l[1]
        new_landmarks[count].z = l[2]
        count = count+1
    #print(new_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x)
    #print(new_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    #print(new_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z)

    return new_landmarks, np.degrees(y_angle)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_angle_shift(a, b, c, d):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # Mid
    d = np.array(d)  # End

    difference = c - b

    a = a + difference

    radians = np.arctan2(d[1] - c[1], d[0] - c[0]) - np.arctan2(a[1] - c[1], a[0] - c[0])
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
    #print(angle)
    if angle > 180.0:
        angle = 360- angle

    return angle

def calculate_angle_3D_shift(a, b, c, d):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # Mid
    d = np.array(d)  # End

    difference = c - b

    a = a + difference

    p1 = a - c
    p2 = d - c
    cosangle = p1.dot(p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    angle = np.abs(np.arccos(cosangle) * 180.0 / np.pi)

    # x2 = c[0] - b[0]
    # y2 = c[1] - b[1]
    # radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # angle = np.abs(radians * 180.0 / np.pi)
    #print(angle)
    if angle > 180.0:
        angle = 360- angle

    return angle
    

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    dist = np.linalg.norm(a - b)

    return dist

def landmark_array(landmarks):
    # Get coordinates

    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    heel_left = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    thumb_left = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    heel_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    thumb_right = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z]

    #la = [[foot_left], [foot_right], [ankle_left], [ankle_right], [knee_left], [knee_right], [hip_left], [hip_right],
    #                  [shoulder_left], [shoulder_right], [elbow_left], [elbow_right], [wrist_left], [wrist_right]]

    la = [[knee_left], [knee_right], [hip_left], [hip_right], [shoulder_left], [shoulder_right]]

    return la

    

def calculate_all_features(landmarks):
    # Get coordinates
    
    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    heel_left = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    thumb_left = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    heel_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    thumb_right = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]

    #print(landmarks)

    features = np.zeros(18)

    features[0] = calculate_angle(foot_left, ankle_left, knee_left)
    features[1] = calculate_angle(ankle_left, knee_left, hip_left)
    features[2] = calculate_angle(knee_left, hip_left, shoulder_left)
    features[3] = calculate_angle(hip_left, shoulder_left, elbow_left)
    features[4] = calculate_angle(shoulder_left, elbow_left, wrist_left)
    features[5] = calculate_angle(elbow_left, wrist_left, thumb_left)
    features[6] = calculate_angle(elbow_left, shoulder_left, shoulder_right)
    features[7] = calculate_angle(knee_left, hip_left, hip_right)

    features[8] = calculate_angle(foot_left, knee_left, hip_left, shoulder_left)
    features[9] = calculate_angle(hip_left, shoulder_left, elbow_left, wrist_left)
    features[10] = calculate_angle(wrist_left, elbow_left, shoulder_left, shoulder_right)
    features[11] = calculate_angle(foot_left, knee_left, hip_left, hip_right)

    features[12] = calculate_angle(foot_right, ankle_right, knee_right)
    features[13] = calculate_angle(ankle_right, knee_right, hip_right)
    features[14] = calculate_angle(knee_right, hip_right, shoulder_right)
    features[15] = calculate_angle(hip_right, shoulder_right, elbow_right)
    features[16] = calculate_angle(shoulder_right, elbow_right, wrist_right)
    features[17] = calculate_angle(elbow_right, wrist_right, thumb_right)
    features[18] = calculate_angle(elbow_right, shoulder_right, shoulder_left)
    features[19] = calculate_angle(knee_right, hip_right, hip_left)

    features[20] = calculate_angle(foot_right, knee_right, hip_right, shoulder_right)
    features[21] = calculate_angle(hip_right, shoulder_right, elbow_right, wrist_right)
    features[22] = calculate_angle(wrist_right, elbow_right, shoulder_right, shoulder_left)
    features[23] = calculate_angle(foot_right, knee_right, hip_right, hip_left)

    # Custom Distances

    features[24] = calculate_distance(ankle_left, ankle_right)
    features[25] = calculate_distance(knee_left, knee_right)
    features[26] = calculate_distance(elbow_left, elbow_right)
    features[27] = calculate_distance(wrist_left, wrist_right)
    features[28] = calculate_distance(wrist_left, ankle_right)
    features[29] = calculate_distance(wrist_left, knee_right)
    features[30] = calculate_distance(wrist_left, hip_right)
    features[31] = calculate_distance(wrist_left, shoulder_right)
    features[32] = calculate_distance(wrist_right, ankle_left)
    features[33] = calculate_distance(wrist_right, knee_left)
    features[34] = calculate_distance(wrist_right, hip_left)
    features[35] = calculate_distance(wrist_right, shoulder_left)
    
    #os.system('clear')
    #print(features)

    return features

def calculate_squat_features(landmarks):
    # Get coordinates
    
    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    heel_left = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    thumb_left = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    heel_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    thumb_right = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z]

    shoulder_origin_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            1,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]

    shoulder_origin_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             1,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

    eye_left = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z]
    eye_right = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z]
    ear_left = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z]
    ear_right = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z]


    #print(landmarks)

    features = np.zeros(14)

    # Foot/Shin Angle
    features[0] = calculate_angle_3D_shift(foot_left, heel_left, ankle_left, knee_left)
    features[1] = calculate_angle_3D_shift(foot_right, heel_right, ankle_right, knee_right)

    # Shin/Femur Angle
    features[2] = calculate_angle_3D(ankle_left, knee_left, hip_left)
    features[3] = calculate_angle_3D(ankle_right, knee_right, hip_right)

    # Femur/Torso Angle
    features[4] = calculate_angle_3D(knee_left, hip_left, shoulder_left)
    features[5] = calculate_angle_3D(knee_right, hip_right, shoulder_right)

    # Back Angle Angle
    #features[6] = calculate_angle_3D(shoulder_origin_left, hip_left, shoulder_left)
    #features[7] = calculate_angle_3D(shoulder_origin_right, hip_right, shoulder_right)

    # Foot/Back Angle
    features[6] = calculate_angle_3D_shift(foot_left, heel_left, hip_left, shoulder_left)
    features[7] = calculate_angle_3D_shift(foot_right, heel_right, hip_right, shoulder_right)

    # Foot/Femur Angle
    features[8] = calculate_angle_3D_shift(foot_left, heel_left, knee_left, hip_left)
    features[9] = calculate_angle_3D_shift(foot_right, heel_right, knee_right, hip_right)

    # Ankle Width
    features[10] = calculate_distance(ankle_left, ankle_right)

    # Foot Angle
    features[11] = calculate_angle_3D_shift(foot_left, heel_left, heel_right, foot_right)

    # Face Angle
    features[12] = calculate_angle_3D_shift(foot_left, heel_left, ear_left, eye_left)
    features[13] = calculate_angle_3D_shift(foot_right, heel_right, ear_right, eye_right)

    #os.system('clear')
    #print(features)

    return features

def calculate_chair_pose_features(landmarks):
    # Get coordinates
    
    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    heel_left = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    thumb_left = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    heel_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    thumb_right = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z]



    #print(landmarks)

    features = np.zeros(13)

    # Foot/Shin Angle
    features[0] = calculate_angle_3D_shift(foot_left, heel_left, ankle_left, knee_left)
    features[1] = calculate_angle_3D_shift(foot_right, heel_right, ankle_right, knee_right)

    # Shin/Femur Angle
    features[2] = calculate_angle_3D(ankle_left, knee_left, hip_left)
    features[3] = calculate_angle_3D(ankle_right, knee_right, hip_right)

    # Femur/Torso Angle
    features[4] = calculate_angle_3D(knee_left, hip_left, shoulder_left)
    features[5] = calculate_angle_3D(knee_right, hip_right, shoulder_right)

    # Torso/Arm Angle
    features[6] = calculate_angle_3D(hip_left, shoulder_left, elbow_left)
    features[7] = calculate_angle_3D(hip_right, shoulder_right, elbow_right)

    # Arm/Forearm Angle
    features[8] = calculate_angle_3D(shoulder_left, elbow_left, wrist_left)
    features[9] = calculate_angle_3D(shoulder_right, elbow_right, wrist_right)

    hw = calculate_distance(hip_left, hip_right)

    # Ankle Width
    features[10] = calculate_distance(ankle_left, ankle_right)/hw

    # Knee Width
    features[11] = calculate_distance(knee_left, knee_right)/hw

    # Wrist Width
    features[12] = calculate_distance(wrist_left, wrist_right)/hw



    #os.system('clear')
    #print('Feature List')
    #print(features)

    return features

########################################################################
## SET WINDOW SIZE
########################################################################
Window.size = (1440, 3120)
Window.fullscreen = True

########################################################################
## LIVE CAMERA CLASS
########################################################################

class LivePoseDetect(Image):
    def __init__(self, **kwargs):
        super(LivePoseDetect, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 60)
        # Clock.schedule_interval(self.update_bbox, 1.50)
        self.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.allow_stretch=True
        self.status_left = None
        self.status_right = None
        self.counter_left = 0
        self.counter_right = 0
        self.status_squat = 'up'
        self.repetitions_count = 0

        self.time_count = 0

        self.chair_pose_status = False
        self.chair_pose_angle_time = 1
        self.chair_pose_feet_time = 1
        self.chair_pose_arm_time = 1
        self.chair_pose_bend_time = 3

        self.images = []
        self.kp_num = 20
        self.temp_kps = []
        self.stop = False
        self.stop_thresh = 10
        
        self.start = False
        self.end = False
        self.face = None

        #Load image
        ret, self.image = self.capture.read()
        self.frame_height = self.image.shape[0]
        self.frame_width = self.image.shape[1]
        self.area = self.frame_height * self.frame_width
        self.ref = time.time_ns()
        self.fps = 0
        self.bbox = (0,0, self.frame_width, self.frame_height)
        self.app_close = False
        self.thread = threading.Thread(target=self.update_bbox)
        self.thread.start()
        self.reset_button = None
        self.save_button = None

        print(f'IDS: {self.ids}')
        # Start a new thread with an infinite loop and stop the current one.
        # print(self.bowling_screen)

    def update_bbox(self):
        while not self.app_close:
            box = process_yolo(self.image)
            tmp = (0,0, self.frame_width, self.frame_height)
            if box is not None:
                
            
                x, y, width, height = box
                bx, by, bwidth, bheight = self.bbox


                distance = np.sqrt(((x + width / 2) - (bx + bwidth / 2)) **2 + ((y + height /2) - (by + bheight/2)))
                area = width * height
                iou = valid_box(box, self.bbox)
                ratio = area/self.area

                print(f'Distance: {distance}\tArea: {area/self.area}\t', end = '')
                
                if iou > 0:
                    print("IOU")
                else:
                    print("NOT")

                if (self.bbox == tmp and ratio >= 0.25) or distance < 50:
                    self.bbox = box


                # # if box is somewhere
                # self.bbox = box
            time.sleep(0.01)

    #Drawing method to execute at intervals
    def update(self, dt):

        #print(MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name)

        if MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name != 'live':
            return
        elif self.reset_button is None:
            self.reset_button = MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_bowling_screen.ids.reset_button
            self.reset_button.bind(on_release=self.reset)

            self.save_button = MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_bowling_screen.ids.save_button
            self.save_button.bind(on_release=self.save_video)
        #Load image
        ret, frame = self.capture.read()
        
        
        # print(self.fps)
        # web camera
        #frame = cv2.flip(frame, 2)

        # monitor 3
        # frame = np.array(sct.grab(monitor))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Recolor image to RGB
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        
                
        self.id = MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_exercise_carousel.index
        try:
            if self.id == 0:
                # Recolor back to BGR
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                (px, py, pw, ph) = self.bbox
                crop = frame[py:py+ph, px:px+pw]

                kps = process_movenet_cropped(crop, self.bbox)
                # print(kps)
                # print(f'Hips Visible: {hips_visible(kps, self.image)}\tFace: {self.face}')

                if hips_visible(kps, self.image):
                    self.temp_kps.append(kps)
                    if not self.end:
                        self.images.append(self.image)

                if len(self.temp_kps) >= self.kp_num:
                    
                    hip_variance = self.get_hip_variance()
                    self.temp_kps.pop(0)
                    if not self.stop:
                        # calculate hip variance
                        # print(f'Hip Variance: {hip_variance}')
                        face = check_face(self.temp_kps[-1])
                        if hip_variance < self.stop_thresh and not face:
                            self.face = face
                            
                            self.stop = True

                        else:
                            self.images.pop(0)
                    
                    elif not self.start and hip_variance > self.stop_thresh:
                        self.start = True
                        self.ref = time.time()
                    elif not self.end and self.start:
                        # new_face = check_face(self.temp_kps[-1])
                        # print(f'New Face: {new_face}')
                        
                        flipped = True

                        for kps in self.temp_kps:
                            flipped = flipped and (self.face != check_face(kps))
                            print(f'{flipped}, ', end='')
                        print()
                        self.end = flipped
                        if self.end:
                            t = time.time() - self.ref
                            print(f"Frames: {len(self.images)}")
                            print(f"TIME: {time.time() - self.ref}")
                            self.fps = 0.8 * len(self.images) / (time.time() - self.ref)




                    self.update_buttons()

                # if len(self.temp_kps) <= self.kp_num and hips_visible(kps):
                #     self.temp_kps.append(kps)
                # if len(self.temp_kps) > self.kp_num:
                #     self.temp_kps.pop(0)
                #     print()

                self.image = draw_keypoints_and_bbox(frame, self.bbox, kps)

                
            elif self.id == 1:
                # Make detection
                results = self.pose.process(self.image)

                # Recolor back to BGR
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                pose_landmarks = results.pose_world_landmarks.landmark

                mp_drawing.draw_landmarks(
                self.image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

                features = calculate_chair_pose_features(pose_landmarks)

                pose_landmarks_norm, pose_angle = calculate_new_landmarks(pose_landmarks)
                print(pose_angle)

                # # 160 degrees is parallel, .13 ideal .07-.19  using actual location values

                squat_depth = (features[2]+features[3]+features[4]+features[5])/4
                print(f'squat depth: {squat_depth}')
                squat_depth = (120-squat_depth)/60 * 100
                if squat_depth < 0:
                    squat_depth = 0
                if squat_depth > 100:
                    squat_depth = 100

                
                arm_angle = (features[6]+features[7])/2
                
                arm_angle = (arm_angle)/180 * 100
                if arm_angle < 0:
                    arm_angle = 0
                if arm_angle > 100:
                    arm_angle = 100

                knee_width = (features[10]+features[11])/2
                
                knee_width = (knee_width-0.75)/1.5 * 100
                if knee_width < 0:
                    knee_width = 0
                if knee_width > 100:
                    knee_width = 100

                features = calculate_squat_features(pose_landmarks)
                features_rw = calculate_squat_features(pose_landmarks)
            
                if (features_rw[2] < 100 or features_rw[3] < 100) & (features_rw[4] < 100 or features_rw[5] < 100):
                    if self.status_squat == 'up':
                        self.repetitions_count = self.repetitions_count + 1
                    self.status_squat = 'down'
                else:
                    self.status_squat = 'up'

                back_angle_left = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)/np.pi*180

                back_angle_right = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/np.pi*180

                back_angle = np.abs(back_angle_left + back_angle_right) / 2

                #print(knee_width)

                back_angle = (back_angle-8) / 45 * 100
                # print(back_angle_l)
                # print(back_angle_r)
                print(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_HIP.value].z)
                if back_angle < 0:
                    back_angle = 0
                if back_angle > 100:
                    back_angle = 100

                head_angle = (features[12]+features[13])/2
                print (head_angle)

                if (features[2] < 90 or features[3] < 90) & (features[4] < 90 or features[5] < 90):
                   if self.status_squat == 'up':
                       self.repetitions_count = self.repetitions_count + 1
                   self.status_squat = 'down'
                else:
                   self.status_squat = 'up'

                # MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.squat_depth_progress_bar.value = int(squat_depth)
                # MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.arm_direction_progress_bar.value = int(arm_angle)
                # MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.knee_width_progress_bar.value = int(knee_width)

                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.squat_progress_bar.value = int(squat_depth)
                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.squat_count.text = str(self.repetitions_count)
                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.back_angle_progress_bar.value = int(back_angle)
                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.knee_width_progress_bar.value = int(knee_width)
                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.head_angle_progress_bar.value = int(head_angle)
                MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_squat_screen.ids.arm_angle_progress_bar.value = int(arm_angle)


                if MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_chair_pose_manager.current == 'cpangle':
                
                    hip_left = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                print(hip_left)
            
                self.image = cv2.circle(self.image, (np.intc(hip_left[0]*frame_width), np.intc(hip_left[1]*frame_height)),radius = 3, color = calculate_bgr(squat_depth), thickness=5)


            elif self.id == 2:
                pass
            
            # pose_landmarks = results.pose_world_landmarks.landmark
            
            

            

            # run inference

            # visualize inference


            
            
           

            




            
            
        except:
            print("HERE")

        
        ## Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        #Convert to Kivy Texture
        buf = cv2.flip(self.image, 0).tobytes()
        texture = Texture.create(size=(self.image.shape[1], self.image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

    def get_hip_variance(self):
        kps = np.array(self.temp_kps)
        right_hip = kps[:, 11, :2]
        left_hip = kps[:,12,:2]
        midpoint = (right_hip + left_hip) / 2.0
        midpoint[:,0] = np.ones((20))
        midpoint[:10,0] = 2 * np.ones((10))
        var = np.var(midpoint, axis=0)
        return np.linalg.norm(var)
    
    def update_buttons(self):
        bowling_screen = MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_bowling_screen
        stop_button = bowling_screen.ids.stop_button
        start_button = bowling_screen.ids.start_button
        end_button = bowling_screen.ids.end_button

        if self.stop:
            # print("Stopped", end='\t')
            stop_button.md_bg_color = (0, 1.0, 0, 1.0)
        
        if self.start:
            # print("Start", end='\t')
            start_button.md_bg_color = (0, 1.0, 0, 1.0)

        
        if self.end:
            # print("End", end='\t')
            end_button.md_bg_color = (0, 1.0, 0, 1.0)
        # print()
    
    def reset(self, dt):
        bowling_screen = MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.live_exercise_screen_manager.ids.live_bowling_screen
        stop_button = bowling_screen.ids.stop_button
        start_button = bowling_screen.ids.start_button
        end_button = bowling_screen.ids.end_button

        self.stop = False
        self.start = False
        self.end = False

        stop_button.md_bg_color = (0.5, 0.5, 0.5, 1.0)
        start_button.md_bg_color = (0.5, 0.5, 0.5, 1.0)
        end_button.md_bg_color = (0.5, 0.5, 0.5, 1.0)

        self.images = []
        self.temp_kps = []
        
        self.bbox = (0, 0, self.frame_width, self.frame_height)

    def save_video(self, dt):
        if self.stop and self.start and self.end:
            height, width, layers = self.images[0].shape
            now = datetime.now()
            date_time_string = now.strftime("%Y-%m-%d_%H:%M:%S")
            video = cv2.VideoWriter("videos/" + date_time_string + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))

            for image in self.images:
                video.write(image)

            video.release()
            print(f"Video saved as temp_video")


    
#def chair_pose_draw(self, dt):

class LiveScreen(Screen):
    pass

########################################################################
## VIDEO CLASS
########################################################################

class VideoPoseDetect(Image):
    def __init__(self, **kwargs):
        super(VideoPoseDetect, self).__init__(**kwargs)
        #Connect to 0th camera
        #self.capture = cv2.VideoCapture('/home/vlb/Theus/APTSystemv1/assets/0918_squat_000005.mp4')
        
        self.capture = 'None'

        #Set drawing interval
        #Clock.schedule_interval(self.update, 1.0 / 60)
        self.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8)

        self.allow_stretch=True
        self.counter_left = 0
        self.counter_right = 0
        self.status_squat = 'up'
        self.repetitions_count = 0
        self.update_event = None

        self.video_length = None
        self.current_frame = 0
        self.play_flag = False
        self.play_once = False

        self.x_axis = []
        self.y_axis = []        
        self.fig = plt.figure()

        self.pose_analysis = PoseAnalysis()

        self.view_squat_depth = False

        #Clock.schedule_once(, 1)

    def file_load(self, filename):
        try:
            self.capture = cv2.VideoCapture(filename)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
            self.current_frame = 0

            #thread = threading.Thread(target=self.process_video, args=())
            #thread.daemon = True
            #thread.start()
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 60)
            self.play_flag = True

            self.counter_left = 0
            self.counter_right = 0
            self.status_squat = 'up'
            self.repetitions_count = 0
            self.update_event = None
        except:
            print('File not video')
        
        # Check if video has process file
        # Load the landmarks

    def process_video(self):

        length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initial image load
        ret, frame = self.capture.read()

        frame_count = 0
        landmarks = []

        while ret:

            # web camera
            #frame = cv2.flip(frame, 2)

            # monitor 3
            # frame = np.array(sct.grab(monitor))

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            # Recolor image to RGB
            self.image = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image.flags.writeable = False

            # Make detection
            results = self.pose.process(self.image)
             
            try:
                landmarks.append(results.pose_world_landmarks.landmark)
                frame_count = frame_count + 1
                ret, frame = self.capture.read()
                print('Percentage: ' + str(frame_count/length * 100), flush=True)
                
            except:
                print('Cannot Process Frame, skipping ' + str(frame_count), flush=True)
                frame_count = frame_count + 1
                ret, frame = self.capture.read()
        
    def load_landmarks(self):
        print('loaded')

    #Drawing method to execute at intervals
    def update(self, dt):

        if MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name != 'video':
            return

        if self.play_flag is False and MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.active is False and self.play_once is False:
            return

        #print(self.current_frame)
        #print(self.video_length)

        if self.current_frame == self.video_length:
            return
       
        #print(app_screen_manager.ids)

        if MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.active:
            self.current_frame = int( MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.value)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
            
        else:
             MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.value = self.current_frame


        self.current_frame = self.current_frame+1

        #Load image
        ret, frame = self.capture.read()

        # web camera
        #frame = cv2.flip(frame, 2)

        # monitor 3
        # frame = np.array(sct.grab(monitor))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Recolor image to RGB
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False

        # Make detection
        results = self.pose.process(self.image)

        # Recolor back to BGR
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
       
        #pose_landmarks = results.pose_landmarks.landmark
        #pose_landmarks_rw = results.pose_world_landmarks.landmark

        #plt.ion()
      

        #l_a = np.squeeze(np.array(landmark_array(pose_landmarks_rw)))
        #print(l_a.shape)

        #self.ax = self.fig.add_subplot(111, projection='3d')

        #self.ax.scatter(l_a[0:2,0], l_a[0:2,1], l_a[0:2,2], c='r', marker='o')
        #self.ax.scatter(l_a[2:4,0], l_a[2:4,1], l_a[2:4,2], c='g', marker='o')
        #self.ax.scatter(l_a[4:6,0], l_a[4:6,1], l_a[4:6,2], c='b', marker='o')
        #self.fig.canvas.draw_idle()
        #plt.pause(0.1)
        #plt.clf()
    

        ############################ left side
        # Extract landmarks
        try:
            pose_landmarks = results.pose_landmarks.landmark
            pose_landmarks_rw = results.pose_world_landmarks.landmark

            mp_drawing.draw_landmarks(
                self.image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            
            features = calculate_squat_features(pose_landmarks)
            features_rw = calculate_squat_features(pose_landmarks_rw)
           
            if (features_rw[2] < 100 or features_rw[3] < 100) & (features_rw[4] < 100 or features_rw[5] < 100):
                if self.status_squat == 'up':
                    self.repetitions_count = self.repetitions_count + 1
                self.status_squat = 'down'
            else:
                self.status_squat = 'up'



            # 160 degrees is parallel, .13 ideal .07-.19  using actual location values
            #print(features_rw[4])
            #print(features_rw[5])
            #print(features_n[6])
            #print(features_n[7])
            #print(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)

            pose_landmarks_norm, hip_angle = calculate_new_landmarks(pose_landmarks_rw)

            #print(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            #print(pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)


            squat_depth_left = (0.25-pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_KNEE.value].y)/0.4 * 100
            squat_depth_right = (0.25-pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)/0.4 * 100
            squat_depth = (squat_depth_left + squat_depth_right) / 2
            if squat_depth < 0:
                squat_depth = 0
            if squat_depth > 100:
                squat_depth = 100

            #features_n = calculate_squat_features(pose_landmarks_norm)
            
            back_angle_left = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)/np.pi*180

            back_angle_right = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/np.pi*180

            back_angle = np.abs(back_angle_left + back_angle_right) / 2

            #print(back_angle)

            back_angle = (back_angle-8) / 45 * 100
            #print(back_angle_l)
            #print(back_angle_r)
            #print(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_HIP.value].z)
            if back_angle < 0:
                back_angle = 0
            if back_angle > 100:
                back_angle = 100


            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.squat_progress_bar.value = int(squat_depth)
            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.squat_count.text = str(self.repetitions_count)
            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.back_angle_progress_bar.value = int(back_angle)

            features_rw = calculate_chair_pose_features(pose_landmarks_rw)

            #self.x_axis = [self.x_axis, float(features_rw[2])]
            #self.y_axis = [self.y_axis, float(features_rw[4])]
            #plt.plot(self.x_axis, self.y_axis)
            #plt.show()

            #self.x_axis.append(features_rw[2])
            #self.y_axis.append(features_rw[4])
            
            #self.fig.canvas.flush_events()


            #self.pose_analysis.add_data(results, features)

            hip_left = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            hip_right = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            print(self.view_squat_depth)

            if self.view_squat_depth == True:
                self.image = cv2.circle(self.image, (np.intc(hip_left[0]*frame_width), np.intc(hip_left[1]*frame_height)),radius = 15, color = calculate_bgr_half(squat_depth), thickness=40)
                self.image = cv2.circle(self.image, (np.intc(hip_right[0]*frame_width), np.intc(hip_right[1]*frame_height)),radius = 15, color = calculate_bgr_half(squat_depth), thickness=40)
            
        

        except:
            pass

        
        ## Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        #Convert to Kivy Texture
        buf = cv2.flip(self.image, 0).tobytes()
        texture = Texture.create(size=(self.image.shape[1], self.image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture
        self.play_once = False

    def squat_update(self, dt):
        pass

    def chair_pose_update(self, results, dt):

        # Extract landmarks
        try:
            pose_landmarks = results.pose_landmarks.landmark
            pose_landmarks_rw = results.pose_world_landmarks.landmark
            

            features = calculate_squat_features(pose_landmarks)
            features_rw = calculate_squat_features(pose_landmarks_rw)
           
            if (features_rw[2] < 100 or features_rw[3] < 100) & (features_rw[4] < 100 or features_rw[5] < 100):
                if self.status_squat == 'up':
                    self.repetitions_count = self.repetitions_count + 1
                self.status_squat = 'down'
            else:
                self.status_squat = 'up'

            

            # 160 degrees is parallel, .13 ideal .07-.19  using actual location values
            #print(features_rw[4])
            #print(features_rw[5])
            #print(features_n[6])
            #print(features_n[7])
            #print(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)

            pose_landmarks_norm = calculate_new_landmarks(pose_landmarks_rw)

      
            print(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            #print(pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)

            squat_depth_left = (0.25-pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_KNEE.value].y)/0.4 * 100
            squat_depth_right = (0.25-pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)/0.4 * 100
            squat_depth = (squat_depth_left + squat_depth_right) / 2
            if squat_depth < 0:
                squat_depth = 0
            if squat_depth > 100:
                squat_depth = 100

            #features_n = calculate_squat_features(pose_landmarks_norm)
            
            back_angle_left = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)/np.pi*180

            back_angle_right = np.arctan(pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x/pose_landmarks_norm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)/np.pi*180

            back_angle = np.abs(back_angle_left + back_angle_right) / 2

            #print(back_angle)

            back_angle = (back_angle-8) / 45 * 100
            #print(back_angle_l)
            #print(back_angle_r)
            #print(pose_landmarks_norm[mp_pose.PoseLandmark.LEFT_HIP.value].z)
            if back_angle < 0:
                back_angle = 0
            if back_angle > 100:
                back_angle = 100

            

            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.squat_progress_bar.value = int(squat_depth)
            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.squat_count.text = str(self.repetitions_count)
            MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.ids.exercisecarousel.ids.squatscreen.back_angle_progress_bar.value = int(back_angle)

            #self.x_axis = [self.x_axis, float(features_rw[2])]
            #self.y_axis = [self.y_axis, float(features_rw[4])]
            #plt.plot(self.x_axis, self.y_axis)
            #plt.show()

            #self.x_axis.append(features_rw[2])
            #self.y_axis.append(features_rw[4])
            
            #self.fig.canvas.flush_events()
        except:
            pass

class VideoScreen(Screen):
    def build(self):
        self.manager_open = False
        #self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
        self.manager = MDFileManager(exit_manager=self.exit_manager, select_path=self.select_path)
        #self.current_dir = self.manager.current_path
        self.path = None
        #self.manager.add_widget(self.file_manager)

        self.ids.folder_button.on_release = self.file_manager_open
        self.ids.restart_button.on_release = self.restart_button_action
        self.ids.play_pause_button.on_release = self.play_button_action
        self.ids.fast_forward_button.on_release = self.forward_button_action


        self.video_update_event = None
        #self.ids.video_slider.max = 0
        #self.ids.video_slider.min = 0
        #self.ids.squat_progress_bar.max = 100
        #self.ids.squat_progress_bar.min = 0


    def file_manager_open(self):
        #if not self.manager:
            #self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
            
            #self.manager.add_widget(self.file_manager)
        self.manager.show(self.manager.current_path)  # output manager to the screen
        self.manager_open = True
        #self.manager.open()

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        toast(path)
        self.path = path
        if self.ids.video_pose_detect.update_event is not None:
            self.ids.video_pose_detect.update_event.cancel()
            self.ids.video_pose_detect.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.ids.video_pose_detect.file_load(self.path)
        self.ids.video_slider.max = int(self.ids.video_pose_detect.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        #self.manager.dismiss()
        self.manager.close()
        self.manager_open = False

    def play_button_action(self):
        self.ids.video_pose_detect.play_flag = not self.ids.video_pose_detect.play_flag
    def restart_button_action(self):
        self.ids.video_pose_detect.current_frame = 0
        self.ids.video_pose_detect.capture.set(cv2.CAP_PROP_POS_FRAMES,self.ids.video_pose_detect.current_frame)
        self.ids.video_pose_detect.play_once = True
    def forward_button_action(self):
        self.ids.video_pose_detect.current_frame = self.ids.video_pose_detect.current_frame + 30
        self.ids.video_pose_detect.capture.set(cv2.CAP_PROP_POS_FRAMES,self.ids.video_pose_detect.current_frame)
        self.ids.video_pose_detect.play_once = True
        

class RightCheckbox(IRightBodyTouch, MDCheckbox):
    pass

class ListItemWithCheckbox(OneLineAvatarIconListItem):
    def __init__(self, **kwargs):
        super(ListItemWithCheckbox, self).__init__(**kwargs)
        self.add_widget(RightCheckbox())

class HomeScreen(Screen):
    pass
    #def build(self):
        #self.pt_search = MDExpansionPanel(panel_cls=MDExpansionPanelOneLine(text='Physical Training'), icon='arm-flex')
        #self.pt_list = MDList()
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Bicep Curl'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Jumping Jack'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Lunge'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Push-Up'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Squat'))
        #self.pt_search.content=self.pt_list
        #self.ids.exercise_view.add_widget(self.pt_search)

        #self.pt_search = MDExpansionPanel(panel_cls=MDExpansionPanelOneLine(text='Yoga'), icon='yoga')
        #self.pt_list = MDList()
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Chair'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Downward Dog'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Tree'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Warrior I'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Warrior II'))
        #self.pt_list.add_widget(ListItemWithCheckbox(text='Warrior III'))
        #self.pt_search.content=self.pt_list
        #self.ids.exercise_view.add_widget(self.pt_search)

        #self.ids.squat_card.size(200, 100)

class UserScreen(Screen):
    pass

class BowlScreen(Screen):
    pass

class SquatScreen(Screen):
    pass

class ChairPoseScreen(Screen):
    pass
class ChairPoseAngleScreen(Screen):
    pass
class ChairPoseFeetScreen(Screen):
    pass
class ChairPoseArmScreen(Screen):
    pass
class ChairPoseBendScreen(Screen):
    pass

class ExerciseCarousel(MDCarousel):
    pass

class LiveExerciseScreenManager(ScreenManager):
    pass

class MainScreen(Screen):
    pass

class SplashScreen(Screen):
    def build(self):
        self.on_enter= self.ids.progress.start()

class EditSliders(BoxLayout):
    pass
########################################################################
## MAIN CLASS
########################################################################
class MainApp(MDApp):
    # Global screen manager variable
    global root
    
    ########################################################################
    ## Build Function
    ########################################################################
    def build(self):
        # Set App Title
        self.title="APT App"
        # Set App Theme
        self.theme_cls.primary_palette='BlueGray'
        self.theme_cls.theme_style = "Dark"
        
        # Load kv screen files to builder
        self.root = Builder.load_file(os.path.join(dirname, 'mainScreen.kv'))
        #print(screen_manager.screen_names)

        # Other Builds
        #self.root.ids.mainscreen.ids.homescreen.build()
        self.root.ids.mainscreen.ids.videoscreen.build()
        self.root.ids.splashscreen.build()
        
        root = self.root
        Window.fullscreen = False

        
        
        # Return screen manager
        return self.root
    ########################################################################
    ## This function runs on app start
    ########################################################################
    def on_start(self):
        # Delay time for splash screen before transitioning to main screen
        Clock.schedule_once(self.change_screen, 6) # Delay for 10 seconds

    def change_window(self, win_name):
        current_window = win_name
        
    ########################################################################
    ## This function changes the current screen to main screen
    ########################################################################
    def change_screen(self, dt):    
        self.root.current = 'MainScreen'

    def eye_squat_depth(self):
        if self.root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.sd_eye.icon == 'eye':
            self.root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.sd_eye.icon = 'eye-outline'
            self.root.ids.mainscreen.ids.videoscreen.ids.video_pose_detect.view_squat_depth = False
        else:
            self.root.ids.mainscreen.ids.videoscreen.ids.exercisecarousel.ids.squatscreen.ids.sd_eye.icon = 'eye'
            self.root.ids.mainscreen.ids.videoscreen.ids.video_pose_detect.view_squat_depth = True

    def squat_depth_settings(self):
        dialog = MDDialog(
            title="Edit Squat Depth Settings",
            type="custom",
            content_cls=EditSliders(),
            buttons=[
                MDFlatButton(
                    text="Done",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                ),
            ],
        )
        dialog.open()
        
        
########################################################################
## RUN APP
########################################################################      
if __name__ == '__main__':
    main = MainApp()
    main.run()
    print()
    main.root.ids.mainscreen.ids.livescreen.ids.live_pose_detect.app_close = True
    print()
########################################################################
## END ==>
########################################################################


















































