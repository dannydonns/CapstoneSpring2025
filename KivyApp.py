# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:27:20 2024

@author: apkun
"""

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.core.image import Texture
from kivy.core.window import Window
from collections import deque

# Load the MoveNet Thunder model
model_name = "movenet_thunder"
if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256

keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


class PoseWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(0)  # Open webcam
        Clock.schedule_interval(self.update, 1 / 30)  # Update at 30 FPS
        self.keypoint_percentages = deque(maxlen=10)

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return
    
        # Preprocess frame for MoveNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        image_resized = tf.image.resize_with_pad(np.expand_dims(rgb_frame, axis=0), input_size, input_size)
        input_image = tf.cast(image_resized, dtype=tf.int32)
    
        # Run inference
        outputs = module.signatures['serving_default'](input_image)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    
        # Transform keypoints for the rotated frame
        keypoints[:, :2] = np.column_stack((
            keypoints[:, 1] * w,  # Scale X to original width
            keypoints[:, 0] * h  # Scale Y to original height
        ))
    
        # Flip Y-coordinate for Kivy's coordinate system
        keypoints[:, 1] = h - keypoints[:, 1]
    
        # Scale keypoints to fit the Kivy window
        scale_x = Window.width / w
        scale_y = Window.height / h
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
    
        # Calculate percentage of detected keypoints
        detected = sum(1 for kp in keypoints if kp[2] > 0.3)
        self.keypoint_percentages.append(detected / len(keypoints) * 100)
    
        # Render video frame on Kivy texture
        buf = cv2.flip(frame, 0).tobytes()  # Flip vertically for Kivy's coordinate system
        texture = Texture.create(size=(w, h), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    
        with self.canvas:
            self.canvas.clear()
            Rectangle(texture=texture, pos=(0, 0), size=Window.size)
    
            # Draw keypoints and connections
            for kp in keypoints:
                x, y, confidence = kp
                if confidence > 0.3:  # Draw only if confidence is above threshold
                    Color(1, 0, 0)
                    Ellipse(pos=(x - 5, y - 5), size=(10, 10))
    
            for connection in connections:
                kp1, kp2 = connection
                if keypoints[kp1][2] > 0.3 and keypoints[kp2][2] > 0.3:
                    x1, y1 = keypoints[kp1][:2]
                    x2, y2 = keypoints[kp2][:2]
                    Color(0, 1, 0)
                    Line(points=[x1, y1, x2, y2], width=2)



    def on_stop(self):
        self.cap.release()


class PoseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")
        self.pose_widget = PoseWidget()
        layout.add_widget(self.pose_widget)

        # Bottom row: graph + button
        bottom_row = BoxLayout(size_hint=(1, 0.25))

        # Graph widget
        self.graph_widget = Widget(size_hint=(0.5, 1))
        bottom_row.add_widget(self.graph_widget)

        # Exit button
        exit_button = Button(text="Exit", size_hint=(0.5, 1))
        exit_button.bind(on_press=self.go_home)
        bottom_row.add_widget(exit_button)

        layout.add_widget(bottom_row)
        self.add_widget(layout)

        Clock.schedule_interval(self.update_graph, 1 / 30)

    def update_graph(self, dt):
        if not self.pose_widget.keypoint_percentages:
            return
    
        with self.graph_widget.canvas:
            self.graph_widget.canvas.clear()
    
            # Draw graph background
            Color(1, 1, 1)  # White background
            Rectangle(pos=self.graph_widget.pos, size=self.graph_widget.size)
    
            # Draw X and Y grid lines
            Color(0, 0, 0)  # Black lines
            for i in range(11):  # X-axis
                x = self.graph_widget.x + i * self.graph_widget.width / 10
                Line(points=[x, self.graph_widget.y, x, self.graph_widget.y + self.graph_widget.height], width=1)
    
            for i in range(5):  # Y-axis
                y = self.graph_widget.y + i * self.graph_widget.height / 4
                Line(points=[self.graph_widget.x, y, self.graph_widget.x + self.graph_widget.width, y], width=1)
    
            # Draw line graph
            Color(0, 0, 1)  # Blue line
            x_step = self.graph_widget.width / 10
            y_scale = self.graph_widget.height / 100
            points = [
                (self.graph_widget.x + i * x_step,
                 self.graph_widget.y + value * y_scale)
                for i, value in enumerate(self.pose_widget.keypoint_percentages)
            ]
            if len(points) > 1:
                Line(points=sum(points, ()), width=2)


    def go_home(self, instance):
        self.manager.current = 'home'


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.0, 0.0, 0.5, 1)  # Navy blue background
            self.bg_rect = Rectangle(size=Window.size, pos=self.pos)
        self.bind(size=self.update_bg, pos=self.update_bg)

        layout = BoxLayout(orientation="vertical", spacing=20, padding=50)
        layout.add_widget(Image(source=r"C:\Users\apkun\OneDrive\Pictures\GingerBreadnought.jpeg", allow_stretch=True))
        start_button = Button(text="Begin Tracking", size_hint=(1, 0.2))
        start_button.bind(on_press=self.start_tracking)
        layout.add_widget(start_button)
        self.add_widget(layout)

    def start_tracking(self, instance):
        self.manager.current = 'pose'

    def update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


class MoveNetApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(PoseScreen(name='pose'))
        return sm


if __name__ == '__main__':
    MoveNetApp().run()

