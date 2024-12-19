"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import math

import cv2
import mediapipe as mp


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return allHands, img

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmlist = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
        return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()

        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            print(lmList1)
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

            # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                      scale=10)

            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingersUp(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")

                # Calculate distance between the index fingers of both hands and draw it on the image
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                          scale=10)

            print(" ")  # New line for better readability of the printed output

        # Display the image in a window
        cv2.imshow("Image", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


import face_recognition
import cv2
import numpy as np
import mysql.connector
from mysql.connector import pooling
from HandsGestureDetector import HandDetector as hd
import time
from Home_Page.home_page import CVApp
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from Virtual_Keyboard.virtual_keyboard import VirtualKeyboard
import os


class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'piyush@12345'),
            'database': 'face_recognition_db'
        }
        self.init_database()
        self.create_connection_pool()

    def init_database(self):
        try:
            conn = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            cursor.execute(f"USE {self.db_config['database']}")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    face_encoding LONGBLOB
                )
            """)
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error initializing database: {err}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def create_connection_pool(self):
        try:
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                **self.db_config
            )
        except mysql.connector.Error as err:
            print(f"Error creating connection pool: {err}")

    def get_connection(self):
        return self.pool.get_connection()

    def insert_user(self, name, face_encoding):
        query = "INSERT INTO users (name, face_encoding) VALUES (%s, %s)"
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
        with self.get_connection() as conn:
            with conn.cursor(prepared=True) as cursor:
                try:
                    cursor.execute(query, (name, face_encoding_bytes))
                    conn.commit()
                    return True
                except mysql.connector.Error as err:
                    print(f"Error inserting user: {err}")
                    conn.rollback()
                    return False

    def get_all_users(self):
        query = "SELECT * FROM users"
        with self.get_connection() as conn:
            with conn.cursor(prepared=True) as cursor:
                try:
                    cursor.execute(query)
                    return cursor.fetchall()
                except mysql.connector.Error as err:
                    print(f"Error fetching users: {err}")
                    return None

    def login_user(self, face_encoding):
        users = self.get_all_users()
        if not users:
            print('Didnt got any user from the database')
            return None

        for user in users:
            if user[2] is not None:  # Check if face_encoding is not None
                stored_encoding = np.frombuffer(user[2], dtype=np.float64)
                # Use a lower tolerance for stricter matching
                if face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.5)[0]:
                    return user
        print('Didnt match any user from the database')
        return None


class Button:
    def __init__(self, text, pos, size):
        self.text = text
        self.pos = pos
        self.size = size

    def draw(self, frame, button_color, text_color):
        cv2.rectangle(frame, self.pos, (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), button_color,
                      cv2.FILLED)
        cv2.putText(frame, self.text, (self.pos[0] + 20, self.pos[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    def is_over(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.size[0] and self.pos[1] < y < self.pos[1] + self.size[1]


class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.randn() * 2
        self.vy = np.random.randn() * 2
        self.radius = np.random.randint(3, 7)
        self.color = tuple(np.random.randint(0, 255, 3).tolist())
        self.life = np.random.randint(20, 40)
        self.alive = True

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        if self.life <= 0:
            self.alive = False

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)


class FaceRecognitionSystem:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.app = CVApp()
        self.vir_Key = VirtualKeyboard()
        self.vir_Key.cam = self.cap

        self.detector = hd()
        self.text = ''
        self.countdown_start = 0
        self.is_counting_down = False
        self.action_after_countdown = None

        self.face_encoding_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_encoding = False

        self.max_encoding_attempts = 5
        self.current_encoding_attempt = 0

        self.bg_color = (245, 230, 200)  # Light beige background
        self.button_color = (70, 150, 180)  # Teal buttons
        self.text_color = (50, 50, 50)  # Dark gray text
        self.highlight_color = (255, 170, 50)  # Orange highlight

        # UI elements
        self.login_button = Button("Login", (100, 300), (300, 80))
        self.signup_button = Button("Signup", (500, 300), (300, 80))
        self.close_button = Button("Close", (1050, 50), (110, 80))
        self.particles = []

    def draw_ui(self, frame):
        title = "Face Recognition System"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.rectangle(frame, (title_x - 10, 70), (title_x + title_size[0] + 10, 110), (0, 0, 0), -1)
        cv2.putText(frame, title, (title_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Draw buttons
        self.login_button.draw(frame, self.button_color, self.text_color)
        self.signup_button.draw(frame, self.button_color, self.text_color)
        self.close_button.draw(frame, (200, 50, 50), (255, 255, 255))  # Red button with white text

        # Draw particles
        for particle in self.particles:
            particle.update()
            particle.draw(frame)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.alive]

        # Draw status text on a black rectangle
        if self.text:
            text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (50, 500), (60 + text_size[0], 530 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(frame, self.text, (55, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def handle_interaction(self, x, y):
        if self.login_button.is_over(x, y):
            self.start_countdown('login')
            self.create_particles(x, y)
        elif self.signup_button.is_over(x, y):
            self.initiate_signup()
            self.create_particles(x, y)
        elif self.close_button.is_over(x, y):
            return True  # Signal to terminate the process
        return False

    def create_particles(self, x, y):
        for _ in range(20):
            self.particles.append(Particle(x, y))

    def draw_button(self, img: np.ndarray, text: str, pos: Tuple[int, int], size: Tuple[int, int]) -> None:
        cv2.rectangle(img, pos, (pos[0] + size[0], pos[1] + size[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, text, (pos[0] + 10, pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def initiate_signup(self) -> None:
        name = self.vir_Key.enter_name(self.cap)
        if not name:
            self.text = "No name entered"
            return
        self.signup_name = name
        self.text = f"User {name} added. Look at the camera for face capture."
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_NORMAL)
        self.start_countdown('signup')

    def encode_face_async(self, frame: np.ndarray) -> None:
        if not self.is_encoding:
            self.is_encoding = True
            self.thread_pool.submit(self._encode_face, frame)

    def _encode_face(self, frame: np.ndarray) -> None:
        if frame is None:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if not face_locations:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        try:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
            self.face_encoding_queue.put(face_encoding)
        except Exception as e:
            print(f"Error encoding face: {str(e)}")
            self.face_encoding_queue.put(None)
        self.is_encoding = False

    def signup(self, name, face_encoding):
        if self.db_manager.insert_user(name, face_encoding):
            return f'User {name} added successfully'
        return f'Failed to add user {name}'

    def login(self, face_encoding):
        user = self.db_manager.login_user(face_encoding)
        if user:
            return f"Welcome back, {user[1]}!"
        return "Face not recognized. Please sign up to continue."

    def start_countdown(self, action: str) -> None:
        self.countdown_start = time.time()
        self.is_counting_down = True
        self.action_after_countdown = action
        self.current_encoding_attempt = 0

    def process_countdown(self, frame: np.ndarray) -> None:
        if self.is_counting_down:
            elapsed_time = time.time() - self.countdown_start
            countdown_num = 3 - int(elapsed_time)
            if countdown_num >= 0:
                cv2.putText(frame, str(countdown_num), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
                self.encode_face_async(frame)
            else:
                self.is_counting_down = False
                if self.action_after_countdown == 'signup':
                    self.complete_signup()
                elif self.action_after_countdown == 'login':
                    self.complete_login()

    def complete_signup(self) -> None:
        face_encoding = self.get_face_encoding()
        if face_encoding is not None:
            result = self.signup(self.signup_name, face_encoding)
            self.text = f"{result}. Get ready to enter CV desktop in 3 seconds."
            self.start_countdown('enter_cv_desktop')

    def complete_login(self) -> None:
        face_encoding = self.get_face_encoding()
        if face_encoding is not None:
            login_result = self.login(face_encoding)
            if "Welcome back" in login_result:
                self.text = f"{login_result} Get ready to enter CV desktop in 3 seconds."
                self.start_countdown('enter_cv_desktop')
            else:
                self.text = login_result
        else:
            self.text = "Face not detected. Please try again."

    def get_face_encoding(self) -> Optional[np.ndarray]:
        while self.current_encoding_attempt < self.max_encoding_attempts:
            try:
                face_encoding = self.face_encoding_queue.get(timeout=1)
                if face_encoding is not None:
                    return face_encoding
                # self.current_encoding_attempt += 1
            except queue.Empty:
                self.current_encoding_attempt += 1

        return None

    def reset_system(self):
        self.text = ''
        self.countdown_start = 0
        self.is_counting_down = False
        self.action_after_countdown = None

        self.face_encoding_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_encoding = False

        self.max_encoding_attempts = 5
        self.current_encoding_attempt = 0

        self.button_color = (70, 150, 180)  # Teal buttons
        self.text_color = (255, 255, 255)  # Dark gray text
        self.highlight_color = (255, 170, 50)  # Orange highlight

        # UI elements
        self.login_button = Button("Login", (100, 300), (300, 80))
        self.signup_button = Button("Signup", (500, 300), (300, 80))
        self.particles = []

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.draw_ui(frame)

            if not self.is_counting_down:
                hands, _ = self.detector.findHands(frame, flipType=False)
                if hands:
                    lmlist = self.detector.findPosition(frame)
                    fingers = self.detector.fingersUp(hands[0])
                    if fingers[1] and not fingers[2]:
                        if self.handle_interaction(lmlist[8][1], lmlist[8][2]):
                            break  # Terminate the process if close button is clicked

            self.process_countdown(frame)

            cv2.imshow('Face Recognition System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition System',
                                                                          cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.action_after_countdown == 'enter_cv_desktop' and not self.is_counting_down:
                time.sleep(1)
                cv2.destroyAllWindows()
                self.app.run(self.cap)
                self.action_after_countdown = None
                self.reset_system()

        self.cap.release()
        cv2.destroyAllWindows()
        self.thread_pool.shutdown()


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()

import cv2
import mediapipe as mp
import time
import math

class poseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                       model_complexity=1,
                                       smooth_landmarks=self.smooth,
                                       enable_segmentation=False,
                                       smooth_segmentation=True,
                                       min_detection_confidence=self.detectionCon,
                                       min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import time
from HandsGestureDetector import HandDetector
from Fitness_Tracker.PoseModule import poseDetector

class ArmCurlsCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = poseDetector()
        self.count = 0
        self.dir = 0
        self.pTime = 0
        self.hands_Detector = HandDetector()
        self.active_arm = 'right'  # Default to right arm
        self.button_left = {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 100}
        self.button_right = {'x1': 250, 'y1': 50, 'x2': 400, 'y2': 100}
        self.f = 0
        self.last_switch_time = 0
        self.switch_delay = 1.0

    def process_frame(self, img):
        img = self.detector.findPose(img, False)
        lmList = self.detector.findPosition(img, False)
        if len(lmList) != 0:
            if self.active_arm == 'right':
                shoulder, elbow, wrist = 15, 13, 11
                angle = self.detector.findAngle(img, shoulder, elbow, wrist)
            else:
                shoulder, elbow, wrist = 12, 14, 16
                angle = self.detector.findAngle(img, shoulder, elbow, wrist)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            color = self.update_count(per)
            self.draw_ui(img, per, bar, color)
        return img

    def update_count(self, per):
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if self.dir == 0:
                self.count += 0.5
                self.dir = 1
        if per == 0:
            color = (0, 255, 0)
            if self.dir == 1:
                self.count += 0.5
                self.dir = 0
        return color

    def draw_ui(self, img, per, bar, color):
        # Progress bar
        cv2.rectangle(img, (1100, 100), (1175, 650), (200, 200, 200), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1120, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Count display
        cv2.rectangle(img, (0, 450), (250, 720), (245, 117, 16), cv2.FILLED)
        cv2.putText(img, str(int(self.count)), (45, 670), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 25)
        cv2.putText(img, "REPS", (40, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

        # Arm selection buttons
        self.draw_button(img, self.button_left, 'Left Arm', self.active_arm == 'left')
        self.draw_button(img, self.button_right, 'Right Arm', self.active_arm == 'right')

        if time.time() - self.last_switch_time < self.switch_delay:
            cv2.putText(img, "Switching...", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def draw_button(self, img, button, text, is_active):
        color = (0, 255, 0) if is_active else (200, 200, 200)
        cv2.rectangle(img, (button['x1'], button['y1']), (button['x2'], button['y2']), color, cv2.FILLED)
        cv2.putText(img, text, (button['x1'] + 10, button['y1'] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def show_fps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def handle_click(self, lmlist, fingers):
        current_time = time.time()
        switch_arm = False

        if self.button_left['x1'] < lmlist[8][1] < self.button_left['x2'] and self.button_left['y1'] < lmlist[8][2] < self.button_left['y2']:
            switch_arm = True
        elif self.button_right['x1'] < lmlist[8][1] < self.button_right['x2'] and self.button_right['y1'] < lmlist[8][2] < self.button_right['y2']:
            switch_arm = True
        elif fingers == [1, 0, 0, 0, 1] or fingers == [0, 0, 0, 0, 1]:
            switch_arm = True

        if switch_arm and (current_time - self.last_switch_time) >= self.switch_delay:
            self.f += 1
            self.last_switch_time = current_time
            if self.f % 2:
                self.active_arm = 'left'
            else:
                self.active_arm = 'right'

        if 900 < lmlist[8][1] < 1000 and 50 < lmlist[8][2] < 90:
            return 1

    def draw_rectangle_with_text(self, image, top_left, bottom_right, text):
        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), -1)

        # Add a border around the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)

        # Calculate the position for the text
        text_position = (top_left[0] + 10, top_left[1] + 30)

        # Draw the text with a shadow
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, text, (text_position[0] + 2, text_position[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        return image

    def run(self, cam=None):
        if cam:
            self.cap = cam
        cv2.namedWindow("Arm Curls Counter")

        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            hands, img = self.hands_Detector.findHands(img, draw=True, flipType=False)
            if hands:
                lmlist = self.hands_Detector.findPosition(img)
                if lmlist:
                    fingers = self.hands_Detector.fingersUp(hands[0])
                    if self.handle_click(lmlist, fingers):
                        cv2.destroyAllWindows()
                        break
            img = self.process_frame(img)
            img = self.show_fps(img)
            img = self.draw_rectangle_with_text(img, (900, 50),(1000, 90), 'BACK')
            cv2.imshow("Arm Curls Counter", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    arm_curls_counter = ArmCurlsCounter()
    arm_curls_counter.run()

import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector as hd
from Volume_Control.volume_control import VolumeControl
from Pain_App.paint_app import VirtualPainter
from Presentation_App.presentation_app import PresentationController
from Pong_Game.pong_app import PongGame
from Virtual_Mouse.virtual_mouse import VirtualMouse
from Math_AI.math_AI_app import HandGestureAI
from Virtual_Keyboard.virtual_keyboard import VirtualKeyboard
from Fitness_Tracker.fitness_tracker import ArmCurlsCounter
import os

class CVApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'

        self.detector = hd(maxHands=1)
        self.vol_control = VolumeControl()
        self.vir_paint = VirtualPainter()
        self.present_app = PresentationController()
        self.vir_mouse = VirtualMouse()
        self.pong_game = PongGame()
        self.math_ai = HandGestureAI(api_key)
        self.vir_keyboard = VirtualKeyboard()
        self.fit = ArmCurlsCounter()

        self.over = False
        self.show_options = False

        self.icon_img = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MenuIcon2.png', cv2.IMREAD_UNCHANGED)
        self.icon_img = cv2.resize(self.icon_img, (40, 40))

    def loading(self):
        start_time = time.time()
        load_duration = 3  # Duration in seconds
        window_created = False

        while time.time() - start_time < load_duration:
            ret, img = self.cam.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)

            progress = int((time.time() - start_time) / load_duration * 800) + 150
            angle = (time.time() - start_time) * 2 * np.pi  # Full rotation every second

            # Draw loading bar
            cv2.rectangle(img, (150, 300), (950, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (150, 300), (progress, 400), (0, 255, 0), -1)
            cv2.rectangle(img, (progress, 295), (progress + 10, 405), (0, 255, 255), -1)

            # Draw rotating circle
            center = (550, 250)
            radius = 30
            circle_x = int(center[0] + radius * np.cos(angle))
            circle_y = int(center[1] + radius * np.sin(angle))
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            cv2.circle(img, (circle_x, circle_y), 10, (255, 0, 0), -1)

            # Display loading text
            cv2.putText(img, "Loading...", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Loading', img)
            cv2.waitKey(1)

            # Set flag when window is created
            if not window_created:
                window_created = True

        # Destroy the 'Loading' window only if it was created
        if window_created:
            cv2.destroyWindow('Loading')

    def draw_interface(self, img):
        buttons = [
            (100, 100, 200, 200, 'Volume Control', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Volume+.png'),
            (300, 100, 400, 200, 'Paint App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PaintApp.png'),
            (500, 100, 600, 200, 'Math AI App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MathAI.png'),
            (700, 100, 800, 200, 'Presentation App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PresentationLogo.png'),
            (100, 300, 200, 400, 'Pong Game', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PONGGame.png'),
            (300, 300, 400, 400, 'Virtual Mouse', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\VirtualMouse.png'),
            (500, 300, 600, 400, 'Fitness Tracker', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\FitnessTracker.png')
        ]

        for (x1, y1, x2, y2, label, icon_path) in buttons:
            # Check if the icon file exists
            if os.path.exists(icon_path):
                # Load the icon image
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

                if icon is not None:
                    # Resize the icon to fit the button
                    icon = cv2.resize(icon, (x2 - x1, y2 - y1))

                    # If the icon has an alpha channel (transparency)
                    if icon.shape[2] == 4:
                        # Create a mask from the alpha channel
                        mask = icon[:, :, 3]
                        # Remove the alpha channel from the icon
                        icon = icon[:, :, 0:3]

                        # Create a region of interest (ROI) on the main image
                        roi = img[y1:y2, x1:x2]

                        # Create a mask inverse
                        mask_inv = cv2.bitwise_not(mask)

                        # Black-out the area of icon in ROI
                        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                        # Take only region of icon from icon image
                        icon_fg = cv2.bitwise_and(icon, icon, mask=mask)

                        # Put icon in ROI and modify the main image
                        dst = cv2.add(img_bg, icon_fg)
                        img[y1:y2, x1:x2] = dst
                    else:
                        # If the icon doesn't have an alpha channel, simply copy it to the main image
                        img[y1:y2, x1:x2] = icon
                else:
                    print(f"Failed to load image: {icon_path}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            else:
                print(f"Image file not found: {icon_path}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # Add text label below the icon
            lines = label.split()
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x1, y2 + 30 + i * 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img

    def draw_options(self, img):
        # cv2.rectangle(img, (900, 100), (1250, 400), (50, 50, 50), -1)  # Options background
        # cv2.putText(img, "Create File", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(img, "Delete File", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(img, "Edit File", (920, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        overlay = img.copy()
        img = cv2.addWeighted(overlay, 0.3, np.zeros(img.shape, img.dtype), 0.7, 0)
        cv2.rectangle(overlay, (900, 100), (1250, 300), (50, 50, 50), -1)  # Options background
        cv2.putText(overlay, "Create File", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "LogOut", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def create_file(self, cam):
        cv2.destroyAllWindows()
        self.vir_keyboard.add_content_to_file(cam)

    def overlay_image(self, background, overlay, x, y):
        """
        Overlays an image (overlay) onto another image (background) at the position (x, y).
        """

        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        bg_h, bg_w, bg_channels = background.shape
        ol_h, ol_w, ol_channels = overlay.shape

        # Ensure the overlay is within the bounds of the background image
        if x + ol_w > bg_w or y + ol_h > bg_h:
            raise ValueError("Overlay image goes out of bounds of the background image.")

        # Get the region of interest (ROI) from the background image
        roi = background[y:y + ol_h, x:x + ol_w]

        # Convert overlay image to have an alpha channel if it doesn't already
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # Separate the alpha channel from the overlay image
        overlay_img = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0

        # Blend the ROI and the overlay image
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay_img[:, :, c] * alpha_mask

        # Replace the original ROI in the background image with the blended ROI
        background[y:y + ol_h, x:x + ol_w] = roi

        return background

    def run(self, cam=None):
        if cam:
            self.cam = cam
        while True:
            ret, img = self.cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            hands, img = self.detector.findHands(img, flipType=False)

            if self.show_options:
                img = self.draw_options(img)

            if hands:

                lmlist = self.detector.findPosition(img)
                fingers = self.detector.fingersUp(hands[0])

                x1, y1 = lmlist[8][1], lmlist[8][2]
                x2, y2 = lmlist[12][1], lmlist[12][2]
                # if fingers == [1, 1, 1, 1, 1]:  exit condition
                #     self.over = True

                if fingers[1] and fingers[2]:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
                    if 1100 < x1 < 1120 and 50 < y1 < 70 or 1100 < x2 < 1120 and 50 < y2 < 70:
                        self.show_options = True
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 20, (0, 0, 255), -1)

                    if self.show_options:
                        if 900 < x1 < 1250 or 900 < x2 < 1250:
                            if 100 < y1 < 200:
                                self.create_file(self.cam)
                            elif 200 < y1 < 300:
                                cv2.destroyAllWindows()
                                break
                        else:
                            self.show_options = False

                    if 100 < x1 < 200 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vol_control.run(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vir_paint.draw(self.cam)
                        self.loading()
                    elif 700 < x1 < 800 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.present_app.run(self.cam)
                        self.loading()
                    elif 100 < x1 < 200 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.pong_game.play_game(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vir_mouse.run(self.cam)
                        self.loading()
                    elif 500 < x1 < 600 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.fit.run(self.cam)
                        self.loading()
                    elif 500 < x1 < 600 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.math_ai.run_app(self.cam)
                        self.loading()

            self.draw_interface(img)
            img = self.overlay_image(img, self.icon_img, 1080, 30)

            cv2.imshow('img', img)
            cv2.waitKey(1)
            if self.over:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    app = CVApp()
    app.run()


import cv2
import numpy as np
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector as hd
from PIL import Image
import textwrap

class HandGestureAI:
    def __init__(self, api_key, model_name='gemini-1.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.prev_pos = None
        self.canvas = None
        self.output_text = ''
        self.detector = hd(maxHands=1)
        self.cam = self.initialize_camera()
        self.initialize_genai()
        self.over = False
        self.response_rectangle = (10, 10, 400, 700)

    def initialize_camera(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)
        return cam

    def initialize_genai(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def get_hand_info(self, img):
        hands, img = self.detector.findHands(img, flipType=False)
        if hands:
            hand = hands[0]
            lmList = self.detector.findPosition(img)
            fingers = self.detector.fingersUp(hand)
            return fingers, lmList
        else:
            return None

    def draw(self, info, img):
        fingers, lmList = info
        current_pos = None

        if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            current_pos = lmList[8][1], lmList[8][2]
            if self.prev_pos is None:
                self.prev_pos = current_pos
            cv2.line(self.canvas, self.prev_pos, current_pos, (255, 0, 255), 10)
            self.prev_pos = current_pos
        elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            self.prev_pos = None
            if 900 < lmList[8][1] < 1000 and 50 < lmList[8][2] < 90:
                self.over = True
        elif fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 1]:
            self.canvas = np.zeros_like(img)
            self.output_text = ''
        # elif fingers == [0, 0, 0, 0, 0]:
        #     self.over = True

    def send_to_ai(self, canvas, fingers):
        if fingers == [0, 0, 0, 0, 1] or fingers == [1, 0, 0, 0, 1]:
            resized_canvas = cv2.resize(canvas, (512, 512))
            pil_img = Image.fromarray(resized_canvas)
            response = self.model.generate_content(["solve this math problem: ", pil_img, ". If the question is complex please explain in detail"])
            return response.text
        return ''

    def draw_response_rectangle(self, image):
        x, y, w, h = self.response_rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

        if self.output_text:
            # Split the text into sentences
            sentences = self.output_text.split('.')
            # Remove empty sentences and add the period back
            sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            line_height = 30
            max_width = w - 20  # Maximum width for text, leaving some padding

            current_y = y + 30
            for sentence in sentences:
                # Wrap each sentence
                wrapped_lines = textwrap.wrap(sentence, width=30)  # Adjust width as needed
                for line in wrapped_lines:
                    # Check if we've reached the bottom of the rectangle
                    if current_y + line_height > y + h:
                        break

                    cv2.putText(image, line, (x + 10, current_y), font, font_scale, (255, 255, 255), font_thickness)
                    current_y += line_height

                # Add an extra line break after each sentence
                current_y += line_height // 2

                # Check if we've reached the bottom of the rectangle
                if current_y > y + h:
                    break

    def draw_rectangle_with_text(self, image, top_left, bottom_right, text):
        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), -1)

        # Add a border around the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)

        # Calculate the position for the text
        text_position = (top_left[0] + 10, top_left[1] + 30)

        # Draw the text with a shadow
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, text, (text_position[0] + 2, text_position[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        return image

    def run_app(self, cap=None):
        if cap is not None:
            self.cam = cap
        while True:
            success, img = self.cam.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(img)
            info = self.get_hand_info(img)
            if info:
                self.draw(info, img)
                new_output = self.send_to_ai(self.canvas, info[0])
                if new_output:
                    self.output_text = ''
                    self.output_text = new_output

            combined = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)

            self.draw_response_rectangle(combined)
            combined = self.draw_rectangle_with_text(combined, (900, 50), (1000, 90), 'BACK')

            cv2.imshow('Hand Gesture AI', combined)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'
    hand_gesture_ai = HandGestureAI(api_key)
    hand_gesture_ai.run_app()


import cv2
from cvzone.HandTrackingModule import HandDetector as hd
import os
import numpy as np
import math
import mysql.connector

class VirtualPainter:
    def __init__(self, username=None):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.detector = hd()
        self.username = username

        self.folder = r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Images'
        self.header_images = [cv2.imread(f'{self.folder}/{img}') for img in os.listdir(self.folder)]
        self.header = self.header_images[0]

        self.icon_img = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MenuIcon2.png',
                                   cv2.IMREAD_UNCHANGED)
        self.icon_img = cv2.resize(self.icon_img, (40, 40))

        self.xp, self.yp = 0, 0
        self.brush_thickness = 30
        self.eraser_thickness = 100
        self.color1 = (255, 192, 203)
        self.color2 = self.color3 = (0, 0, 0)
        self.selected = ''
        self.circle_flag = False
        self.done = False
        self.doneL = False
        self.gone = False
        self.line_flag = False
        self.show_options = False
        self.fill_option = ''
        self.lm_list = []
        self.fill_type = None
        self.fill_start_angle = 0
        self.fill_end_angle = 0
        self.canvas_states = []
        self.max_states = 70  # Maximum number of states to store
        self.undo_button_active = False
        self.current_color = self.color1

        self.circle_x1, self.circle_y1, self.radius = 0, 0, 0
        self.line_start, self.line_end = (0, 0), (0, 0)
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)

        self.brush_size = 15
        self.min_brush_size = 5
        self.max_brush_size = 50

        self.slider_center = (130, 180)  # Center position of the circular slider
        self.slider_radius = 50  # Radius of the circular slider

        self.dropdown_button_active = False
        self.dropdown_options = ["Save", "Exit"]
        self.show_menu = False

    def draw_brush_slider(self, img):
        cv2.rectangle(img, (10, 130), (260, 160), (200, 200, 200), -1)
        cv2.rectangle(img, (10, 130), (10 + int(250 * (self.brush_size - self.min_brush_size) / (self.max_brush_size - self.min_brush_size)), 160), (0, 255, 0), -1)
        cv2.putText(img, f"Brush Size: {self.brush_size}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def adjust_brush_size(self, x):
        if 10 <= x <= 260:
            self.brush_size = int(self.min_brush_size + (x - 10) * (self.max_brush_size - self.min_brush_size) / 250)
            self.brush_thickness = self.brush_size

    def save_canvas_state(self):
        if len(self.canvas_states) >= self.max_states:
            self.canvas_states.pop(0)
        self.canvas_states.append(self.img_canvas.copy())
        self.undo_button_active = True

    def draw_menu_button(self, background, overlay, x, y):
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        bg_h, bg_w, bg_channels = background.shape
        ol_h, ol_w, ol_channels = overlay.shape

        # Ensure the overlay is within the bounds of the background image
        if x + ol_w > bg_w or y + ol_h > bg_h:
            raise ValueError("Overlay image goes out of bounds of the background image.")

        # Get the region of interest (ROI) from the background image
        roi = background[y:y + ol_h, x:x + ol_w]

        # Convert overlay image to have an alpha channel if it doesn't already
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # Separate the alpha channel from the overlay image
        overlay_img = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0

        # Blend the ROI and the overlay image
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay_img[:, :, c] * alpha_mask

        # Replace the original ROI in the background image with the blended ROI
        background[y:y + ol_h, x:x + ol_w] = roi

        return background

    def draw(self, cam=None):
        if cam:
            self.cam = cam
        while True:
            _, img = self.cam.read()
            if _:
                img = cv2.flip(img, 1)
                hands, img = self.detector.findHands(img, flipType=False)

                self.draw_undo_button(img)
                self.draw_brush_slider(img)

                if self.show_options:
                    img = self.draw_options(img)

                if hands:
                    self.lm_list = self.detector.findPosition(img)
                    if self.lm_list:
                        if self.process_hand_gestures(img, hands):
                            break

                        if 1000 < self.lm_list[8][1] < 1180 and 50 < self.lm_list[8][2] < 140 and not self.show_menu:
                            self.show_menu = True
                        if self.show_menu:
                            img = self.draw_menu(img)

                        if self.fill_type:
                            cv2.ellipse(img, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                                        0, self.fill_start_angle, self.fill_end_angle, self.color2, 2)

                img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, self.img_canvas)

                img[:104, :1007] = self.header

                img = self.draw_menu_button(img, self.icon_img, 1100, 80)

                cv2.imshow('img', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()

    def process_hand_gestures(self, img, hands):
        x1, y1 = self.lm_list[8][1], self.lm_list[8][2]
        x2, y2 = self.lm_list[12][1:]
        fingers = self.detector.fingersUp(hands[0])

        if fingers[1] and fingers[2]:
            self.xp, self.yp = 0, 0
            if 1090 < x1 < 1180 and 10 < y1 < 60 and self.undo_button_active:
                # print(f"Attempting to undo. Button active: {self.undo_button_active}")
                self.undo()
                cv2.waitKey(10)
            elif 130 < y1 < 160:
                self.adjust_brush_size(x1)
            elif x1 < 1000:
                self.show_menu = False
                self.select_tool(x1, y1, x2, y2, img)
        elif fingers[1] and not fingers[2]:
            if self.show_options:
                self.select_fill_option(x1, y1)
            elif self.fill_type:
                self.select_fill_area(x1, y1, img)
            elif self.show_menu:
                if 1000 < x1 < 1180 and 70 < y1 < 200:
                    # if 100 < y1 < 140:  # Save
                    #     print('save')
                    #     self.save_screenshot(img)
                    if 150 < y1 < 200:  # Exit
                        print('exit')
                        cv2.destroyAllWindows()
                        return 1
            else:
                self.draw_on_canvas(img, hands)
        elif not fingers[1] and not fingers[2]:
            if self.fill_type:
                self.apply_selected_fill()
                self.save_canvas_state()
                self.undo_button_active = True
                # print("Canvas state saved, undo button activated")
            self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

    def save_screenshot(self, img):
        cv2.imwrite(f'Output.jpg', img)

    def draw_menu(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (1000, 120), (1250, 210), (50, 50, 50), -1)  # Options background
        # cv2.putText(overlay, "Save", (1020, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Exit", (1020, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def select_fill_option(self, x, y):
        if 900 < x < 1250:
            if 100 < y < 200:
                # print("Full circle fill selected")
                self.fill_type = "full"
            elif 200 < y < 300:
                # print("Half circle fill selected")
                self.fill_type = "half"
            elif 300 < y < 400:
                # print("Quarter circle fill selected")
                self.fill_type = "quarter"

            if self.fill_type:
                self.show_options = False

    def select_fill_area(self, x, y, img):
        dx = x - self.circle_x1
        dy = self.circle_y1 - y  # Invert y-axis
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        if self.fill_type == "full":
            self.fill_start_angle = 0
            self.fill_end_angle = 360
        elif self.fill_type == "half":
            self.fill_start_angle = angle
            self.fill_end_angle = (angle + 180) % 360
        elif self.fill_type == "quarter":
            self.fill_start_angle = angle
            self.fill_end_angle = (angle + 90) % 360

        # Create a separate mask for the preview
        preview_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(preview_mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                    0, self.fill_start_angle, self.fill_end_angle, 255, -1)

        # Create an overlay with the preview color
        preview_color = np.array(self.color2, dtype=np.uint8)
        overlay = np.full(img.shape, preview_color, dtype=np.uint8)

        # Apply the overlay only to the masked area
        mask_3channel = cv2.merge([preview_mask, preview_mask, preview_mask])
        overlay_area = cv2.bitwise_and(overlay, mask_3channel)
        img_area = cv2.bitwise_and(img, cv2.bitwise_not(mask_3channel))

        # Blend the overlay with the original image
        result = cv2.add(img_area, overlay_area)
        cv2.addWeighted(img, 0.5, result, 0.5, 0, dst=img)

    def apply_selected_fill(self):
        if self.fill_type:
            mask = np.zeros(self.img_canvas.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                        0, self.fill_start_angle, self.fill_end_angle, 255, -1)
            self.img_canvas[mask == 255] = self.color2
            self.fill_type = None  # Reset fill type after applying

    def apply_fill(self, start_angle, end_angle):
        mask = np.zeros(self.img_canvas.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                    0, start_angle, end_angle, 255, -1)
        self.img_canvas[mask == 255] = self.color2

    def draw_options(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (900, 100), (1250, 400), (50, 50, 50), -1)  # Options background
        cv2.putText(overlay, "Fill full circle", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Fill half circle", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Fill quarter circle", (920, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def select_tool(self, x1, y1, x2, y2, img):
        if y1 < 130:
            # self.header = self.header_images[0]
            if 10 < x1 < 100:
                self.header = self.header_images[0]
                self.color1 = (255, 192, 203)
                self.color2 = (0, 0, 255)
                self.color3 = (255, 192, 203)
                self.selected = 'brush1'
            elif 200 < x1 < 300:
                self.header = cv2.resize(self.header_images[1], (1007, 104), interpolation=cv2.INTER_AREA)
                self.color1 = (0, 0, 255)
                self.color2 = (0, 0, 255)
                self.color3 = (0, 0, 255)
                self.selected = 'brush2'
            elif 450 < x1 < 550:
                self.select_circle()
            elif 600 < x1 < 700:
                self.select_line()
            elif 800 < x1 < 900:
                self.color1 = (0, 0, 0)
                self.color2 = (0, 0, 255)
                self.selected = 'eraser'

        cv2.line(img, (x1, y1), (x2, y2), self.color3, 3)

    def select_circle(self):
        self.color2 = (0, 0, 0)
        self.color1 = (0, 0, 0)
        self.color3 = (255, 0, 255)
        self.selected = 'circle'
        self.circle_flag = True
        self.done = False

    def select_line(self):
        self.color2 = (0, 0, 0)
        self.color1 = (0, 0, 0)
        self.color3 = (0, 255, 0)
        self.selected = 'line'
        self.line_flag = True
        self.doneL = False

    def draw_on_canvas(self, img, hands):
        x1, y1 = self.lm_list[8][1], self.lm_list[8][2]
        cv2.circle(img, (x1, y1), 10, (255, 255, 255), -1)

        # Check if the finger was just lowered
        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        # Calculate the distance between current and previous point
        distance = ((x1 - self.xp) ** 2 + (y1 - self.yp) ** 2) ** 0.5

        # If the distance is too large, assume the finger was lifted and reset the previous point
        if distance > 50:  # You may need to adjust this threshold
            self.xp, self.yp = x1, y1

        if self.selected == 'brush1' or self.selected == 'brush2':
            self.draw_line(x1, y1, img)
        elif self.selected == 'eraser':
            self.draw_eraser(x1, y1, img)
        elif self.selected == 'circle':
            self.draw_circle(x1, y1, img, hands)
        elif self.selected == 'line':
            self.draw_line_shape(x1, y1, img, hands)

        self.xp, self.yp = x1, y1

        self.save_canvas_state()
        # self.undo_button_active = True
        # print("Canvas state saved after drawing")

    def draw_undo_button(self, img):
        if self.undo_button_active:
            cv2.rectangle(img, (1090, 10), (1180, 60), (0, 255, 0), -1)
            cv2.putText(img, "Undo", (1105, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(img, (1090, 10), (1180, 60), (200, 200, 200), -1)
            cv2.putText(img, "Undo", (1105, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    def undo(self):
        if len(self.canvas_states) > 0:
            self.img_canvas = self.canvas_states.pop()
        if len(self.canvas_states) == 0:
            self.undo_button_active = False

    def draw_line(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)

    def draw_eraser(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)

    def draw_circle(self, x1, y1, img, hands):
        if len(hands) == 2 and self.circle_flag:
            self.circle_x1, self.circle_y1 = x1, y1
            lm_list2 = self.detector.findPosition(img, 1)
            thumbX, thumbY = lm_list2[8][1], lm_list2[8][2]
            self.radius = int(((thumbX - x1) ** 2 + (thumbY - y1) ** 2) ** 0.5)
            x3, y3 = self.lm_list[4][1], self.lm_list[4][2]
            length = int(((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5)
            if length < 160:
                self.circle_flag = False
                self.done = True
                self.show_options = True
                self.color2 = (255, 0, 0)
                cv2.circle(img, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
                cv2.circle(self.img_canvas, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
        if not self.done:
            cv2.circle(img, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
            cv2.circle(self.img_canvas, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)

    def draw_line_shape(self, x1, y1, img, hands):
        if len(hands) == 2 and self.line_flag:
            self.line_start = (x1, y1)
            lm_list2 = self.detector.findPosition(img, 1)
            self.line_end = (lm_list2[8][1], lm_list2[8][2])
            x3, y3 = self.lm_list[4][1], self.lm_list[4][2]
            length = int(((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5)

            if length < 160:
                self.line_flag = False
                self.doneL = True
                self.color2 = (255, 0, 0)
                cv2.line(img, self.line_start, self.line_end, self.color2, 5)
                cv2.line(self.img_canvas, self.line_start, self.line_end, self.color2, 5)
        if not self.doneL:
            cv2.line(img, self.line_start, self.line_end, self.color2, 5)
            cv2.line(self.img_canvas, self.line_start, self.line_end, self.color2, 5)


if __name__ == '__main__':
    painter = VirtualPainter(username='OM')
    painter.draw()

import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random

class PongGame:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        self.img_background = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\Background.png')
        self.img_game_over = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\gameOver.png')
        self.img_ball = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\Ball.png', cv2.IMREAD_UNCHANGED)
        self.img_bat1 = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\bat1.png', cv2.IMREAD_UNCHANGED)
        self.img_bat2 = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\bat2.png', cv2.IMREAD_UNCHANGED)

        self.ball_pos = [100, 100]
        self.speed_x = 20
        self.speed_y = 20
        self.score = [0, 0]

        self.game_over = False
        self.countdownFlag = False
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)

        # Powerup variables
        self.powerup_active = False
        self.powerup_timer = 0
        self.powerup_hand = None
        self.powerup_timer2 = 0

    def reset(self):
        self.ball_pos = [100, 100]
        self.speed_x = 25
        self.speed_y = 25
        self.score = [0, 0]
        self.game_over = False
        self.countdownFlag = False
        self.powerup_active = False
        self.powerup_timer = 0
        self.powerup_hand = None
        self.powerup_timer2 = 0
        self.img_game_over = cv2.imread(r'/CV_Pong_Game/Resources/gameOver.png')

    def countdown(self, img):
        for i in range(3, 0, -1):
            img_copy = img.copy()
            cv2.putText(img_copy, str(i), (600, 360), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 10)
            cv2.imshow('img', img_copy)
            cv2.waitKey(1000)

    def draw_powerup(self, img):
        if not self.powerup_active and random.randint(0, 120) < 5:  # 5% chance of powerup spawn
            self.powerup_x = random.randint(100, 1100)  # Random X-coordinate for powerup
            self.powerup_y = random.randint(100, 400)  # Random Y-coordinate for powerup
            self.powerup_active = True
            self.powerup_timer = 200  # Powerup duration in frames

        if self.powerup_active:
            radius = 25  # Radius of the powerup circle
            cv2.circle(img, (self.powerup_x, self.powerup_y), radius, (0, 255, 0), -1)  # Draw powerup circle
            if self.powerup_x - radius < self.ball_pos[0] < self.powerup_x + radius and self.powerup_y - radius < self.ball_pos[1] < self.powerup_y + radius:
                self.powerup_active = False
                self.powerup_timer2 = 100
                self.powerup_timer = 0
                if self.ball_pos[0] > 0:
                    self.powerup_hand = 'Right'
                else:
                    self.powerup_hand = 'Left'

        if self.powerup_timer:
            self.powerup_timer -= 1  # Decrement powerup timer
        if self.powerup_timer2:
            self.powerup_timer2 -= 1  # Decrement powerup timer
        else:
            self.powerup_hand = None
        if self.powerup_timer <= 0:
            self.powerup_active = False

        return img

    def draw_bats(self, img, hands):
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = self.img_bat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == 'Left':
                if self.powerup_hand == 'Left':
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1))
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1 + (h1 - 30)))
                    if 59 - 10 < self.ball_pos[0] < 59 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 * 2):
                        self.speed_x *= -1
                        self.ball_pos[0] += 20
                        self.score[0] += 1
                else:
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1))
                    if 59 - 10 < self.ball_pos[0] < 59 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 // 2):
                        self.speed_x *= -1
                        self.ball_pos[0] += 20
                        self.score[0] += 1

            if hand['type'] == 'Right':
                if self.powerup_hand == 'Right':
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1))
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1 + (h1 - 30)))
                    if 1120 < self.ball_pos[0] < 1170 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 * 2):
                        self.speed_x *= -1
                        self.ball_pos[0] -= 20
                        self.score[1] += 1
                else:
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1))
                    if 1120 < self.ball_pos[0] < 1170 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 // 2):
                        self.speed_x *= -1
                        self.ball_pos[0] -= 20
                        self.score[1] += 1

        return img

    def play_game(self, cam):
        self.cam = cam
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)
            img_raw = img.copy()

            hands, img = self.detector.findHands(img, flipType=False)
            img = cv2.addWeighted(img, 0.2, self.img_background, 0.8, 0)

            if not self.countdownFlag:
                self.countdown(img)
                self.countdownFlag = True  # To ensure countdown only happens once

            img = self.draw_powerup(img)

            if hands:
                img = self.draw_bats(img, hands)

            if self.ball_pos[1] >= 500 or self.ball_pos[1] <= 10:
                self.speed_y *= -1

            if self.ball_pos[0] < 10 or self.ball_pos[0] > 1200:
                self.game_over = True

            if self.game_over:
                img = self.img_game_over
                cv2.putText(img, str(max(self.score[0], self.score[1])).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 0, 200), 5)
            else:
                self.ball_pos[0] += self.speed_x
                self.ball_pos[1] += self.speed_y

                cv2.putText(img, str(self.score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
                cv2.putText(img, str(self.score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

                img = cvzone.overlayPNG(img, self.img_ball, self.ball_pos)

            img[580:700, 20:233] = cv2.resize(img_raw, (213, 120))

            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('r'):
                self.reset()
            elif key == ord('q'):
                break

if __name__ == "__main__":
    game = PongGame()
    game.play_game(None)


import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector as hd

class PresentationController:
    def __init__(self, folder=r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Presentations'):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.detector = hd(maxHands=1)
        self.folder = folder
        self.images = os.listdir(folder)
        self.img_number = 0
        self.ws, self.hs = 213, 120
        self.threshold = 425
        self.buttonPressed = False
        self.buttonCounter = 0
        self.buttonDelay = 10
        self.annotations = [[]]
        self.annotationsNumber = -1
        self.annotationsFlag = False
        self.over = False

    def load_image(self):
        current_image = os.path.join(self.folder, self.images[self.img_number])
        img_current = cv2.imread(current_image)
        img_current = cv2.resize(img_current, (1280, 720), interpolation=cv2.INTER_AREA)
        return img_current

    def process_frame(self, img, img_current):
        hands, img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)
        cv2.line(img, (0, self.threshold), (1280, self.threshold), (0, 255, 0), 10)

        if hands:
            fingers = self.detector.fingersUp(hands[0])

        if lmList and not self.buttonPressed:
            cx, cy = hands[0]['center']
            if cy <= self.threshold:
                if fingers[0]:
                    self.change_slide(-1)
                elif fingers[4]:
                    self.change_slide(1)

            x1, y1 = self.map_coordinates(lmList[8][1], lmList[8][2])
            if fingers[1] and fingers[2]:
                self.annotationsFlag = False
                cv2.circle(img_current, (x1, y1), 20, (0, 0, 255), -1)
            elif fingers[1] and not fingers[2]:
                self.draw_annotation(x1, y1, img_current)
            else:
                self.annotationsFlag = False
            if fingers == [0, 1, 1, 1, 0]:
                self.remove_last_annotation()

            if fingers == [1, 1, 1, 1, 1]:
                self.over = True

        if self.buttonPressed:
            self.buttonCounter += 1
            if self.buttonCounter > self.buttonDelay:
                self.buttonCounter = 0
                self.buttonPressed = False

        self.draw_annotations(img_current)
        return img, img_current

    def change_slide(self, direction):
        self.buttonPressed = True
        self.img_number = max(0, min(self.img_number + direction, len(self.images) - 1))
        self.annotations = [[]]
        self.annotationsNumber = -1
        self.annotationsFlag = False

    def map_coordinates(self, x1, y1):
        x1 = int(np.interp(x1, [1280 // 2, 1280], [0, 1280]))
        y1 = int(np.interp(y1, [150, 720 - 150], [0, 720]))
        return x1, y1

    def draw_annotation(self, x1, y1, img_current):
        if not self.annotationsFlag:
            self.annotationsFlag = True
            self.annotationsNumber += 1
            self.annotations.append([])
        cv2.circle(img_current, (x1, y1), 20, (0, 0, 255), -1)
        self.annotations[self.annotationsNumber].append([x1, y1])

    def remove_last_annotation(self):
        if self.annotations:
            self.annotations.pop(-1)
            self.annotationsNumber -= 1
            self.buttonPressed = True

    def draw_annotations(self, img_current):
        for i in range(len(self.annotations)):
            for j in range(len(self.annotations[i])):
                if j:
                    cv2.line(img_current, self.annotations[i][j - 1], self.annotations[i][j], (0, 0, 200), 10)

    def run(self, cam):
        self.cam = cam
        while True:
            _, img = self.cam.read()
            if not _:
                break
            img = cv2.flip(img, 1)
            img_current = self.load_image()
            img, img_current = self.process_frame(img, img_current)
            img_Small = cv2.resize(img, (self.ws, self.hs))
            img_current[:self.hs, 1280 - self.ws:1280] = img_Small
            cv2.imshow('Presentation', img_current)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                self.over = False

                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = PresentationController()
    controller.run()


import cv2
import numpy as np
import cvzone
import time
from PIL import ImageFont, ImageDraw, Image
import os
# from spellchecker import SpellChecker
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector


class VirtualKeyboard:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        self.detector = HandDetector()
        self.keyboard = Controller()

        self.keys_en = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
            ["SHIFT", "SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]
        ]

        self.keys_mr = [
            ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ञ", "ट"],
            ["ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प"],
            ["फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष"],
            ["SHIFT", "SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]
        ]

        self.current_keys = self.keys_en
        self.final_text = ''
        self.prev_len = 0

        font_path = "../Resources/static/NotoSansDevanagari-Regular.ttf"
        if not os.path.isfile(font_path):
            raise FileNotFoundError(
                f"Font file '{font_path}' not found. Please ensure the file exists and the path is correct.")
        self.font = ImageFont.truetype(font_path, 32)

        self.shift = False
        self.caps = False
        self.debounce_time = 1
        self.last_press_time = time.time()

        # self.spell = SpellChecker('en')

        self.button_list = self.create_button_list()

        self.mode = "normal"  # Add this line to track the current mode
        self.save_button = Button([1100, 100], "SAVE", [100, 80])

    def enter_name(self, cam=None):
        self.mode = "name_entry"
        self.final_text = ""
        try:
            if cam:
                self.cam = cam
            while True:
                success, img = self.cam.read()
                img = cv2.flip(img, 1)

                hands, img = self.detector.findHands(img, flipType=False)
                lmlist1 = self.detector.findPosition(img)

                img = self.draw_all(img)

                if len(lmlist1):
                    for button in self.button_list:
                        x, y = button.pos
                        w, h = button.size
                        name = self.check_hand_position(img, button, hands)
                        if name:
                            cv2.destroyWindow('img')
                            return name

                img = draw_text_with_pil(img, f"Enter name: {self.final_text}", (50, 50), self.font, (0, 0, 255))

                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        # finally:
            # cv2.destroyWindow('img')
        return self.final_text.strip()

    def add_content_to_file(self, cam):
        self.mode = "file_entry"
        self.final_text = ""
        if cam:
            self.cam = cam
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)

            hands, img = self.detector.findHands(img, flipType=False)
            lmlist1 = self.detector.findPosition(img)

            img = self.draw_all(img)

            # Draw the save button
            x, y = self.save_button.pos
            w, h = self.save_button.size
            cv2.rectangle(img, self.save_button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
            self.save_button.render_text(img, self.font)

            if len(lmlist1) and hands:
                for button in self.button_list + [self.save_button]:
                    n = self.check_hand_position(img, button, hands)
                    if n:
                        cv2.destroyAllWindows()
                        return

            img = draw_text_with_pil(img, self.final_text, (50, 50), self.font, (0, 0, 255))

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def create_button_list(self):
        button_list = []
        c = 0
        for i in range(len(self.current_keys)):
            for j, key in enumerate(self.current_keys[i]):
                if key in ["SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]:
                    button_list.append(Button([100 * j + 80 + c, 100 * i + 150], key, [100, 80]))
                    c += 30
                elif key in ['SHIFT']:
                    button_list.append(Button([100 * j + 50, 100 * i + 150], key, [100, 80]))
                else:
                    button_list.append(Button([100 * j + 50, 100 * i + 150], key))
        return button_list

    def draw_all(self, img):
        img_new = np.zeros_like(img, np.uint8)
        for button in self.button_list:
            x, y = button.pos
            cvzone.cornerRect(img_new, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
            cv2.rectangle(img_new, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
            button.render_text(img_new, self.font)
        out = cv2.addWeighted(img, 0.5, img_new, 0.5, 0)
        return out

    def handle_button_press(self, button_text):
        if button_text == "LANG":
            self.switch_language()
        elif button_text == "SHIFT":
            self.shift = not self.shift
        elif button_text == "CAPS":
            self.caps = not self.caps
        elif button_text == "BACK":
            self.final_text = self.final_text[:-1]
        elif button_text == "CLEAR":
            self.final_text = ''
            self.prev_len = 0
        elif button_text == "SPACE":
            self.final_text += " "
        elif button_text == "ENTER":
            self.final_text += '\n'
        else:
            if self.shift:
                button_text = button_text.upper()
            elif self.caps:
                button_text = button_text.lower()
            self.final_text += button_text

    def check_hand_position(self, img, button, hands):
        x, y = button.pos
        w, h = button.size
        lmlist1 = self.detector.findPosition(img, 0)
        if len(hands) == 2:
            lmlist2 = self.detector.findPosition(img, 1)
            for lmlist in [lmlist1, lmlist2]:
                if (x < lmlist[8][1] < x + w and y < lmlist[8][2] < y + h) or (
                        x < lmlist[12][1] < x + w and y < lmlist[12][2] < y + h):
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                    l, _, _ = self.detector.findDistance((lmlist[8][1], lmlist[8][2]), (lmlist[12][1], lmlist[12][2]),
                                                         img)

                    if self.mode == 'file_entry':
                        if int(l) < 45:
                            if time.time() - self.last_press_time > self.debounce_time:
                                self.last_press_time = time.time()
                                if button.text == "SAVE":
                                    with open("output.txt", "w", encoding="utf-8") as f:
                                        f.write(self.final_text)
                                    self.mode = "normal"
                                    return 1
                                else:
                                    self.handle_button_press(button.text)

                    if self.mode == 'name_entry':
                        if int(l) < 45:
                            if time.time() - self.last_press_time > self.debounce_time:
                                self.last_press_time = time.time()
                                if button.text == "ENTER":
                                    self.mode = "normal"
                                    return self.final_text.strip()
                                else:
                                    self.handle_button_press(button.text)

                    if int(l) < 45 and time.time() - self.last_press_time > self.debounce_time:
                        self.last_press_time = time.time()
                        self.handle_button_press(button.text)

        else:
            if (x < lmlist1[8][1] < x + w and y < lmlist1[8][2] < y + h) or (
                    x < lmlist1[12][1] < x + w and y < lmlist1[12][2] < y + h):
                cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                l, _, _ = self.detector.findDistance((lmlist1[8][1], lmlist1[8][2]), (lmlist1[12][1], lmlist1[12][2]),
                                                     img)
                if self.mode == 'file_entry':
                    if int(l) < 45:
                        if time.time() - self.last_press_time > self.debounce_time:
                            self.last_press_time = time.time()
                            if button.text == "SAVE":
                                with open("output.txt", "w", encoding="utf-8") as f:
                                    f.write(self.final_text)
                                self.mode = "normal"
                                return 1
                            else:
                                self.handle_button_press(button.text)

                if self.mode == 'name_entry':
                    if int(l) < 45:
                        if time.time() - self.last_press_time > self.debounce_time:
                            self.last_press_time = time.time()
                            if button.text == "ENTER":
                                self.mode = "normal"
                                print(self.final_text)
                                return self.final_text.strip()
                            else:
                                self.handle_button_press(button.text)

                if int(l) < 45 and time.time() - self.last_press_time > self.debounce_time:
                    self.last_press_time = time.time()
                    self.handle_button_press(button.text)

    def switch_language(self):
        self.current_keys = self.keys_mr if self.current_keys == self.keys_en else self.keys_en
        self.button_list = self.create_button_list()

    # def autocorrect(self):
    #     if self.final_text and self.final_text[-1] == " " and self.current_keys == self.keys_en:
    #         words = self.final_text.split()
    #         if len(words) > self.prev_len:
    #             last_word = words[-1]
    #             corrected_word = self.spell.correction(last_word)
    #             if corrected_word != last_word:
    #                 self.prev_len = len(words)
    #                 if self.caps:
    #                     corrected_word = corrected_word.upper()
    #                 words[-1] = corrected_word
    #                 self.final_text = ' '.join(words) + ' '

    def run(self):
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)

            hands, img = self.detector.findHands(img, flipType=False)
            lmlist1 = self.detector.findPosition(img)

            img = self.draw_all(img)

            if len(lmlist1) and hands:

                for button in self.button_list:
                    x, y = button.pos
                    w, h = button.size

                    self.check_hand_position(img, button, hands)

            # self.autocorrect()

            img = draw_text_with_pil(img, self.final_text, (50, 50), self.font, (0, 0, 255))

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


class Button:
    def __init__(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text
        self.rendered_text = None

    def render_text(self, img, font):
        if self.rendered_text is None:
            self.rendered_text = draw_text_with_pil(np.zeros_like(img, np.uint8), self.text,
                                                    (self.pos[0] + 10, self.pos[1] + 20), font, (255, 255, 255))
        img[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]] = cv2.addWeighted(
            img[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]], 0.5,
            self.rendered_text[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]], 0.5, 0
        )


def draw_text_with_pil(img, text, position, font, color):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)


if __name__ == "__main__":
    virtual_keyboard = VirtualKeyboard()
    # virtual_keyboard.run()
    print(virtual_keyboard.add_content_to_file())

import cv2
import numpy as np
from HandsGestureDetector import HandDetector as hd
import time
import pyautogui
import math


class VirtualMouse:
    def __init__(self, wCam=1280, hCam=720, smoothing=10):
        self.wCam = wCam
        self.hCam = hCam
        self.smoothing = smoothing
        self.prevX = 0
        self.prevY = 0
        self.curX = 0
        self.curY = 0
        self.cap = self.initialize_camera()
        self.detector = hd(maxHands=1)
        self.wScr, self.hScr = pyautogui.size()
        self.mode = 'normal'  # Can be 'normal' or 'finger'
        self.cTime = 0
        self.pTime = 0
        self.over = False

    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.wCam)
        cap.set(4, self.hCam)
        return cap

    def set_mode(self, mode):
        if mode in ['normal', 'finger']:
            self.mode = mode

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img)
        self.lmList = self.detector.findPosition(img)

        if len(self.lmList):
            x1, y1 = self.lmList[8][1], self.lmList[8][2]
            x2, y2 = self.lmList[12][1], self.lmList[12][2]

            self.fingers = self.detector.fingersUp(hands[0])
            cv2.rectangle(img, (100, 100), (self.wCam - 50, self.hCam - 50), (255, 255, 0), 3)

            if self.mode == 'normal':
                if self.fingers[1] and not self.fingers[2]:
                    self.move_mouse(x1, y1, img)

                if self.fingers[1] and self.fingers[2]:
                    self.click_mouse(x1, y1, x2, y2, img)
            elif self.mode == 'finger':
                if self.fingers[1] and not self.fingers[2]:
                    self.move_mouse(x1, y1, img, finger_only=True)

                if self.fingers[1] and self.fingers[2]:
                    self.click_mouse(x1, y1, x2, y2, img)

                if self.fingers == [1, 1, 1, 1, 1] or self.fingers == [0, 1, 1, 1, 1]:
                    self.over = True

        self.display_fps(img)
        return img

    def move_mouse(self, x1, y1, img, finger_only=False):
        x3, y3 = np.interp(x1, (100, self.wCam - 100), (0, self.wScr)), np.interp(y1, (100, self.hCam - 100), (0, self.hScr))
        if finger_only:
            # self.curX = self.prevX + (x3 - self.prevX) // self.smoothing
            # self.curY = self.prevY + (y3 - self.prevY) // self.smoothing
            # pyautogui.moveTo(self.curX, self.curY)
            # self.prevX, self.prevY = self.curX, self.curY
            x1, y1 = self.lmList[8][1], self.lmList[8][2]
            # x1 = int(np.interp(x1, [1280 // 2, w], [0, 1280]))
            # y1 = int(np.interp(y1, [150, 720 - 150], [0, 720]))

            if self.fingers[1] and self.fingers[2]:
                cv2.circle(img, (x1, y1), 20, (0, 0, 255), -1)
        else:
            pyautogui.moveTo(x3, y3)
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)

    def click_mouse(self, x1, y1, x2, y2, img):
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 60:
            pyautogui.click()

    def display_fps(self, img):
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    def run(self, cam=None):
        if cam:
            self.cap = cam
        while True:
            success, img = self.cap.read()
            if not success:
                break

            # cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)

            img = self.process_frame(img)
            cv2.imshow('Virtual Mouse', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.over:
                self.over = False

                break
            elif key == ord('n'):
                self.set_mode('normal')
            elif key == ord('f'):
                self.set_mode('finger')
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vm = VirtualMouse()
    vm.run()


import cv2
import time
import numpy as np
import HandsGestureDetector as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeControl:
    def __init__(self, wCam=1280, hCam=720):
        self.cap = self.initialize_camera(wCam, hCam)
        self.detector = htm.HandDetector(maxHands=1)
        self.volume, self.minVolume, self.maxVolume = self.initialize_volume_control()
        self.selected = 1
        self.pTime = 0
        self.volBar = 400
        self.vol = 0
        self.volPer = 0
        self.volbar1 = 150
        self.volbar2 = 157
        self.over = False

    @staticmethod
    def initialize_camera(wCam, hCam):
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(4, hCam)
        return cap

    @staticmethod
    def initialize_volume_control():
        device = AudioUtilities.GetSpeakers()
        interface = device.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        minVolume = volume.GetVolumeRange()[0]
        maxVolume = volume.GetVolumeRange()[1]
        return volume, minVolume, maxVolume

    @staticmethod
    def draw_hand_landmarks(img, lmlist):
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return x1, y1, x2, y2, cx, cy

    @staticmethod
    def update_volume(length, minVolume, maxVolume):
        vol = np.interp(length, [50, 300], [minVolume, maxVolume])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = int(np.interp(length, [50, 300], [0, 100]) // 5) * 5
        return vol, volBar, volPer

    @staticmethod
    def display_options(img):
        cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (300, 100), (400, 200), (0, 0, 0), -1)
        cv2.putText(img, '1', (130, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(img, '2', (330, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    @staticmethod
    def selected_option(fingers, lmList, selected):
        x, y = lmList[8][1], lmList[8][2]

        if fingers[1] and 100 < x < 200 and 100 < y < 200:
            selected = 0
        elif fingers[1] and 300 < x < 400 and 100 < y < 200:
            selected = 1
        return selected

    @staticmethod
    def display_volume_bar(img, volBar, volPer, selected, volbar1, volbar2):
        if not selected:
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 1)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (10, 470), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (150, 300), (950, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (150, 300), (volbar1, 400), (0, 255, 0), -1)
            cv2.rectangle(img, (volbar1, 295), (volbar2, 405), (0, 255, 255), -1)
            cv2.putText(img, f'{0}%', (80, 368), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(img, f'{100}%', (960, 365), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(img, f'{int(volPer)} %', (volbar1 - 25, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    @staticmethod
    def display_fps(img, fps):
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 0, 255), 3)

    # def toggle_mute(self):
    #     current_volume = self.volume.GetMasterVolumeLevel()
    #     if current_volume != self.minVolume:
    #         self.volume.SetMasterVolumeLevel(self.minVolume, None)
    #     else:
    #         self.volume.SetMasterVolumeLevel(self.maxVolume, None)

    # def change_brightness(level):
    #     sbc.set_brightness(level)

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img)
        description = ''
        self.display_options(img)

        if hands:
            lmlist = self.detector.findPosition(img)
            fingers = self.detector.fingersUp(hands[0])

            if fingers == [0, 1, 1, 1, 1]:
                self.over = True

            elif fingers[1] and fingers[2] and fingers[3]:
                description = 'Selection Mode:- You can select any of the two given options to change the volume.'
                self.selected = self.selected_option(fingers, lmlist, self.selected)

            else:
                if not self.selected and fingers[1] and not fingers[2]:
                    description = 'Change Mode 1:- You can use your index finger and thumb to change the volume by increasing or decreasing the distance between them.'
                    x1, y1, x2, y2, cx, cy = self.draw_hand_landmarks(img, lmlist)
                    length = math.hypot(x2 - x1, y2 - y1)
                    self.vol, self.volBar, self.volPer = self.update_volume(length, self.minVolume, self.maxVolume)
                    if fingers[4]:
                        self.volume.SetMasterVolumeLevel(self.vol, None)
                elif self.selected and fingers[1] and fingers[2]:
                    description = 'Change Mode 2:- You can use your index finger and middle finger to change the volume by placing your index finger anywhere inside the rectangle.'
                    x11, y11 = lmlist[8][1], lmlist[8][2]
                    if fingers[1] and fingers[2] and 300 <= y11 <= 400:
                        x11 = int(x11)
                        self.vol = np.interp(self.volbar1, [150, 950], [self.minVolume, self.maxVolume])
                        self.volbar1 = int(np.interp(x11, [150, 950], [150, 950]))
                        self.volbar2 = self.volbar1 + 7
                        self.volPer = int(np.interp(self.volbar1, [150, 950], [0, 100]) // 5) * 5
                        self.volume.SetMasterVolumeLevel(self.vol, None)
                # elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
                #     description = 'Brightness Control Mode:- You can use your index finger and ring finger to change the brightness by increasing or decreasing the distance between them.'
                #     x1, y1, x2, y2, cx, cy = self.draw_hand_landmarks(img, lmlist)
                #     length = math.hypot(x2 - x1)
                #     brightness = np.interp(length, [50, 300], [0, 100])
                #     change_brightness(int(brightness))
                # elif fingers[4] and not any(fingers[:4]):
                #     description = 'Mute/Unmute Mode:- You can use your pinky finger to toggle mute and unmute.'
                #     self.toggle_mute()
                else:
                    description = 'There is nothing assigned to this gesture.'

        # self.draw_text_within_rectangle(img, description, (500, 50), (1100, 200))
        self.display_volume_bar(img, self.volBar, self.volPer, self.selected, self.volbar1, self.volbar2)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        self.display_fps(img, fps)

        return img

    # @staticmethod
    # def draw_text_within_rectangle(image, text, rect_top_left, rect_bottom_right, font=cv2.FONT_HERSHEY_SIMPLEX,
    #                                font_scale=1, font_color=(255, 255, 255), font_thickness=2):
    #     # Define the rectangle
    #     cv2.rectangle(image, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    #
    #     # Calculate the width of the rectangle
    #     rect_width = rect_bottom_right[0] - rect_top_left[0]
    #     rect_height = rect_bottom_right[1] - rect_top_left[1]
    #
    #     # Split text into words
    #     words = text.split(' ')
    #     current_line = ''
    #     y0, dy = rect_top_left[1] + 30, 30
    #
    #     for word in words:
    #         # Check if adding the next word will go out of the rectangle's width
    #         test_line = current_line + word + ' '
    #         text_size, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
    #
    #         if text_size[0] > rect_width:
    #             # Write the current line on the image
    #             cv2.putText(image, current_line, (rect_top_left[0] + 10, y0), font, font_scale, font_color,
    #                         font_thickness, lineType=cv2.LINE_AA)
    #             current_line = word + ' '
    #             y0 += dy
    #
    #             # Check if we are exceeding the height of the rectangle
    #             if (y0 + dy - rect_top_left[1]) > rect_height:
    #                 break
    #         else:
    #             current_line = test_line
    #
    #     # Write the last line
    #     cv2.putText(image, current_line, (rect_top_left[0] + 10, y0), font, font_scale, font_color, font_thickness,
    #                 lineType=cv2.LINE_AA)

    def run(self, cam):
        self.cap = cam
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = self.process_frame(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                self.over = False
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vc = VolumeControl()
    vc.run()
