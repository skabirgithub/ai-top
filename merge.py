import math
import cv2
from pynput import keyboard
import mediapipe as mp
import pyautogui
from deepface import DeepFace
from pynput.keyboard import Key, Controller, Listener
from datetime import datetime
import numpy as np
import sys
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import time
import matplotlib.pyplot as plt
import pandas as pd
import ctypes
# Import the real-time dashboard file
from real_time_dashboard import create_real_time_dashboard
import tkinter as tk
import requests
import json

# Face detection and emotion analysis code

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe's solutions
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize the MediaPipe holistic model
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    # Initialize variables for gaze tracking
    prev_time = 0

    # Video Capture
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        raise IOError("Cannot open Web camera")
    # Define the codec and create a VideoWriter object to save the video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    record_out = cv2.VideoWriter('Record_Videso.mp4', fourcc, 3.0, (640, 480))
    # Initialize lists to hold data

    engagement_count_sk = 0
    bored_count_sk = 0
    frustrated_count_sk = 0
    
    result_trigger = []
    

    time_values = []
    mouse_values = []
    reaction_values = []
    left_eye_values = []
    right_eye_values = []
    keyboard_values = []
    heart_rate_values = []

    array_distg = []


    emotion_median_valuesk = []
    left_eye_interactionsk = []
    right_eye_interactionsk = []
    mouse_interaction_distancesk = []
    keyboard_interaction_scoresk = []
    heart_mean_diffsk = []
    heart_mean_diffsk.append(0)
    keyboard_array = []

    y_data1, y_data2, y_data3, y_data4 = [], [], [], []

    def on_key_press(key):
        append_current_time_to_keyboard_array()

    def append_current_time_to_keyboard_array():
        current_time = time.time()
        keyboard_array.append(current_time)

    # Create a listener for keyboard events
    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    keyboard_listener.start()

    realWidth = 640
    realHeight = 480
    videoWidth = 160
    videoHeight = 120
    videoChannels = 3
    videoFrameRate = 15

    # Webcam Parameters
    webcam = cv2.VideoCapture(0)
    detector = FaceDetector()

    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    # Color Magnification Parameters
    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0

    # Helper Methods
    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    # Output Display Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (30, 40)
    bpmTextLocation = (videoWidth // 2, 40)
    fpsTextLoaction = (500, 600)

    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 1

    # Initialize Gaussian Pyramid
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    # Bandpass Filter for Specified Frequencies
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    # Heart Rate Calculation Variables
    bpmCalculationFrequency = 10  # 15
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i = 0
    ptime = 0
    ftime = 0

    # Real-time Plot Parameters
    # plotBufferSize = 100
    # plotX = np.arange(plotBufferSize)
    # plotY = np.zeros(plotBufferSize)

    # plt.ion()  # Turn on interactive mode for live plot
    # fig, ax = plt.subplots()
    # line, = ax.plot(plotX, plotY)
    # ax.set_ylim(0, 200)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('BPM')
    # ax.set_title('Heart Rate Monitor')

    # Helper Methods
    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    while True:
        if keyboard_array:
            #print("Keyboard Array:", keyboard_array)
            #keyboard_array = []  # Clear the array after printing
            time.sleep(0)

        # Capture frame-by-frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Write the frame to the output video file
        record_out.write(frame)
        # Analyze DeepFace Emotion Detection
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # DeepFace Emotion Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Eye gaze
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark

            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    # Mouse movement with eye and head
                    # pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            # Print Eye Gaze
            #print(left[0].x, left[1].y)

        results = holistic.process(rgb_frame)

        if results.face_landmarks is not None:
            # Get the coordinates of the left and right eye landmarks
            left_eye_landmarks = [
                (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                for landmark in results.face_landmarks.landmark[36:42]
            ]
            right_eye_landmarks = [
                (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                for landmark in results.face_landmarks.landmark[42:48]
            ]

            # Calculate the gaze point by taking the midpoint of the eye landmarks
            left_eye_midpoint = (
                int(sum([x for x, _ in left_eye_landmarks]) / 6),
                int(sum([y for _, y in left_eye_landmarks]) / 6)
            )
            right_eye_midpoint = (
                int(sum([x for x, _ in right_eye_landmarks]) / 6),
                int(sum([y for _, y in right_eye_landmarks]) / 6)
            )

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Font for emotion detection text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Result of emotion detection
        # cv2.putText(frame,
        #             result[0]["dominant_emotion"][:],
        #             (50, 100),
        #             font, 3,
        #             (0, 0, 255),
        #             2,
        #             cv2.LINE_4)
        #---------------------------------------------------------------------------------------------------#
        #heart rate functionality Start
        frame, bboxs = detector.findFaces(frame, draw=False)
        frameDraw = frame.copy()
        ftime = time.time()
        fps = 1 / (ftime - ptime)
        ptime = ftime

        # cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        if bboxs:
            x1, y1, w1, h1 = bboxs[0]['bbox']
            cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
            detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[mask == False] = 0

            # Grab a Pulse
            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

                # Update live plot
                # plotY = np.roll(plotY, -1)
                # plotY[-1] = bpm
                # line.set_ydata(plotY)
                # fig.canvas.draw()
                
            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            # Reconstruct Resulting Frame
            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize
            outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
            frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

            bpm_value = bpmBuffer.mean()
            # cv2.putText(frameDraw, f'BPM: {bpm_value:.2f}', bpmTextLocation, font, fontScale, fontColor, lineType)
            if i > bpmBufferSize:
                # cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', (videoWidth // 2, 40), scale=2)
                if result_trigger[-1] == 'engaged':
                    text_color = (0, 255, 0)  # Green
                elif result_trigger[-1] == 'bored':
                    text_color = (0, 0, 255)  # Blue
                else:
                    text_color = (255, 0, 0)  # Red
                cv2.putText(frameDraw, f'{result_trigger[-1]}', (videoWidth // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness=1, lineType=cv2.LINE_AA)
                
                cv2.putText(frameDraw, f'Travel Mouse:{int(calculate_interaction(mouse_values))},Eye:{int(calculate_eye_interaction(left_eye_values).sum())} px', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            else:
                cvzone.putTextRect(frameDraw, "Initiating ...", (30, 40), scale=2)
                
                
            if len(sys.argv) != 2:
                cv2.imshow("AI Top Full", frameDraw)

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                  #  break
        else:
            cv2.imshow("AI Top Full", frameDraw)
        #---------------------------------------------------------------------------------------------------#
        # Frame window for video/face
        #cv2.imshow('Demo Video', frame)
        # Get the current timestamp
        timestamp = datetime.now()
        #print('Time Domain:', timestamp)

        # Mouse pointer coordinates
        mouse_x, mouse_y = pyautogui.position()

        # Print mouse pointer coordinates
        #print(f"Mouse coordinates: X={mouse_x}, Y={mouse_y}")

        # Print the emotion detection result
        #print('Reaction:', result[0]["dominant_emotion"])
        # Print Eye Gaze
        #print(f"Left eye gaze: {left_eye_midpoint}")
        #print(f"Right eye gaze: {right_eye_midpoint}")
        # Video frame window close
        if cv2.waitKey(1) & 0xFF == 27:
            # 'esc' key was pressed
            break
        time_values.append(timestamp)
        #Convert mouse_x to a string and concatenate it with mouse_y
        mouse_values.append("(" + str(mouse_x) + "," + str(mouse_y) + ")")
        reaction_values.append(result[0]["dominant_emotion"])
        left_eye_values.append(left_eye_midpoint)
        right_eye_values.append(right_eye_midpoint)
        keyboard_values.append(str(Key))
        heart_rate_values.append(bpm_value)

        # old start
        #eye interaction function
        def calculate_eye_interaction(eye_array):
            total_distance = 0.0
            array_dist = []
            for i in range(1, len(eye_array)):
                x_diff = eye_array[i][0] - eye_array[i - 1][0]
                y_diff = eye_array[i][1] - eye_array[i - 1][1]
                distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
                array_dist.append(distance)
                array_distg.append(distance)
                #total_distance += distance
            #return total_distance
            return np.array(array_dist)
        
        left_eye_interaction = calculate_eye_interaction(left_eye_values)
        right_eye_interaction = calculate_eye_interaction(right_eye_values) 
        pd.DataFrame(left_eye_interaction).to_csv("left_eye.csv")
        pd.DataFrame(right_eye_interaction).to_csv("right_eye.csv")

        left_eye_interaction = left_eye_interaction.sum()
        right_eye_interaction = right_eye_interaction.sum()

        # print("Left Eye Array:", left_eye_values)
        # print("Right Eye Array:", right_eye_values)
        # print("Left Eye Interaction:", left_eye_interaction)
        # print("Right Eye Interaction:", right_eye_interaction)

        #mouse interaction function
        def calculate_interaction(mouse_array):
            total_distance = 0.0

            for i in range(1, len(mouse_array)):
                x1, y1 = map(int, mouse_array[i - 1][1:-1].split(','))
                x2, y2 = map(int, mouse_array[i][1:-1].split(','))
                x_diff = x2 - x1
                y_diff = y2 - y1
                distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
                total_distance += distance
                mouse_interaction_distancesk.append(distance)
            return total_distance
        # Call the function to calculate the interaction
        mouse_interaction_distance = calculate_interaction(mouse_values)

        # print("Mouse Array:", mouse_values)
        # print("Mouse Interaction Distance:", mouse_interaction_distance)

        #heart rate interaction function
        def calculate_mean_difference(heart_rate_data):
        # Calculate the differences between consecutive data points
            differences = [heart_rate_data[i] - heart_rate_data[i - 1] for i in range(1, len(heart_rate_data))]

            # Calculate the mean of the differences using numpy.mean()
            mean_difference = np.mean(differences)
            if differences is not None:
                heart_mean_diffsk.append(mean_difference)
            else:
                heart_mean_diffsk.append(0)                       
            return mean_difference

        # Call the function to calculate the median difference
        heart_mean_diff = calculate_mean_difference(heart_rate_values)

        # print("Heart Rate Data:", heart_rate_values)
        # print("Heart Mean Difference:", heart_mean_diff)

        #Emotion interaction function
        def calculate_emotion_median(emotion_array):
            emotion_data = {
                'sad': 37.65260875225067,
                'angry': 0.15512987738475204,
                'surprise': 0.0022171278033056296,
                'fear': 1.2489334680140018,
                'happy': 4.609785228967667,
                'disgust': 9.698561953541684e-07,
                'neutral': 56.33133053779602
            }
            
            emotion_values = [emotion_data.get(emotion, 0) for emotion in emotion_array]
            median_value = np.median(emotion_values)
            
            return median_value

        # Call the function to calculate the median value
        emotion_median_value = calculate_emotion_median(reaction_values)

        # print("Emotion Array:", reaction_values)
        # print("Emotion Median Value:", emotion_median_value)

        #keyboard interaction
        def calculate_interaction_score(keyboard_array):
            if len(keyboard_array) <= 1:
                return 0
            
            time_intervals = [keyboard_array[i] - keyboard_array[i - 1] for i in range(1, len(keyboard_array))]
            average_interval = sum(time_intervals) / len(time_intervals)
            
            interaction_score = 1 / average_interval  # Inverse of average interval (higher interval -> lower score)
            
            return interaction_score

        # Call the function to calculate the interaction score
        keyboard_interaction_score = calculate_interaction_score(keyboard_array)

        # print("Keyboard Array:", keyboard_array)
        # print("Keyboard Interaction Score:", keyboard_interaction_score)

        
        #Engagement Analysis
        def classify_user_engagement_with_weights(left_eye_interaction, right_eye_interaction, mouse_interaction_distance,
                                                emotion_median_value, keyboard_interaction_score,
                                                heart_mean_difference,
                                                engagement_weight, boredom_weight, frustration_weight):
            engagement_threshold = 2500
            boredom_threshold = 2000

            total_interaction = left_eye_interaction + right_eye_interaction
            total_interaction_score = total_interaction * engagement_weight
            boredom_score = mouse_interaction_distance * boredom_weight
            emotion_score = emotion_median_value * 0.2
            keyboard_score = keyboard_interaction_score * 0.1
            heart_score = heart_mean_difference * 0.05  # Adjust weight as needed

            total_score = total_interaction_score + boredom_score + emotion_score + keyboard_score + heart_score

            if total_score > engagement_threshold:
                return "Engaged"
            elif total_score < boredom_threshold:
                return "Bored"
            else:
                return "Frustrated"

        # User-defined weights
        engagement_weight = 0.4
        boredom_weight = 0.3
        frustration_weight = 0.3

        # Call the function to get the classification
        classification = classify_user_engagement_with_weights(left_eye_interaction, right_eye_interaction,
                                                            mouse_interaction_distance, emotion_median_value,
                                                            keyboard_interaction_score, heart_mean_diff,
                                                            engagement_weight, boredom_weight, frustration_weight)

        # print("Result:",classification)
                
        # old end
        # -----------------------------#
        #31-08-2023,abir - start

        def calculate_threshold(data):
            cleaned_data = [value for value in data if not np.isnan(value)]
            data = cleaned_data
            # Extract the last three values from the data array
            last_three_values = data
            if len(data) > 3:
                last_three_values = data[-3:]
            # Calculate the first-order derivative of the last three values
            derivative = np.diff(last_three_values)
            # Calculate the median and standard deviation of the derivative
            median_derivative = np.median(derivative)
            std_derivative = np.std(derivative)
            # Calculate the threshold using median +- 3 * std
            threshold = median_derivative + 3 * std_derivative
            return np.median(threshold)
            
        def calculate_velocity(data):
            # Extract the last three values from the data array
            cleaned_data = [value for value in data if not np.isnan(value)]
            data = cleaned_data
            last_three_values = data
            if len(data) > 3:
                last_three_values = data[-3:]
            # Calculate the first-order derivative of the last three values
            derivative = np.diff(last_three_values)
            return np.mean(derivative)
            

        

        def classify_based_on_threshold(eye_velocity, eye_threshold, mouse_velocity, mouse_threshold, heart_rate, heart_rate_threshold, emotion):
            global engagement_count_sk, bored_count_sk, frustrated_count_sk
            # print("eye_velocity",eye_velocity)
            # print("eye_threshold",eye_threshold)
            # print("mouse_velocity",mouse_velocity)
            # print("mouse_threshold",mouse_threshold)
            # print("heart_rate",heart_rate)
            # print("heart_rate_threshold",heart_rate_threshold)
            # print("myemotion",emotion)

            
            
            if eye_velocity < eye_threshold:
                if mouse_velocity < mouse_threshold:
                    if heart_rate < heart_rate_threshold:
                        if emotion in [4.609785228967667, 56.33133053779602, 0.0022171278033056296]:
                            engagement_count_sk += 1
                            print("Result Engaged")
                            result_trigger.append('Engaged')
                        else:
                            if emotion in [0.15512987738475204, 1.2489334680140018]:
                                frustrated_count_sk += 1
                                print("Result Frustrated")
                                result_trigger.append('Frustrated')
                            else:
                                bored_count_sk += 1
                                print("Result Bored")
                                result_trigger.append('Bored')

            return max(engagement_count_sk, bored_count_sk, frustrated_count_sk)
        

        def estimation_calculation():
            # print("fahadE")
            # emotion_median_valuesk.append(calculate_emotion_median(reaction_values))
            # left_eye_interactionsk.append(calculate_eye_interaction(left_eye_values))
            # right_eye_interactionsk.append(calculate_eye_interaction(right_eye_values)) 
            # mouse_interaction_distancesk.append(calculate_interaction(mouse_values))
            # keyboard_interaction_scoresk.append(calculate_interaction_score(keyboard_array))
            # heart_mean_diffsk.append(calculate_mean_difference(heart_rate_values))

            # print('dsf',array_distg)
            left_eye_threshold_sk = calculate_threshold(array_distg)
            mouse_interaction_threshold_sk = calculate_threshold(mouse_interaction_distancesk)
            # print("vitore",heart_mean_diffsk)
            heart_rate_threshold_sk = calculate_threshold(heart_mean_diffsk)
            # heart_rate_threshold_sk = 0.324

            left_eye_velocity_sk = calculate_velocity(array_distg)
            mouse_interaction_velocity_sk = calculate_velocity(mouse_interaction_distancesk)
            heart_rate_velocity_sk = calculate_velocity(heart_mean_diffsk)



            # print("left_eye_velocity_sk", left_eye_velocity_sk);
            # print("left_eye_threshold_sk", left_eye_threshold_sk);
            # print("mouse_interaction_distancesk", mouse_interaction_distancesk);
            # print("mouse_interaction_velocity_sk", mouse_interaction_velocity_sk);
            # print("mouse_interaction_threshold_sk", mouse_interaction_threshold_sk);
            # print("heart_mean_diffsk", heart_mean_diffsk);
            # print("heart_rate_velocity_sk", heart_rate_velocity_sk);
            # print("heart_rate_threshold_sk", heart_rate_threshold_sk);
            result = classify_based_on_threshold(left_eye_velocity_sk, left_eye_threshold_sk, mouse_interaction_velocity_sk, mouse_interaction_threshold_sk, heart_rate_velocity_sk, heart_rate_threshold_sk, calculate_emotion_median(reaction_values))

            return result;
        estimation_calculation()
        print('engagement_count_sk',engagement_count_sk)
        print('bored_count_sk',bored_count_sk)
        print('frustrated_count_sk',frustrated_count_sk)

        # print("estimation calculation:",result)
        #31-08-2023,abir-end
        
        # if engagement_count_sk > 10:
        #     print("aaaaaaaaaaaaa",array_distg)
    
    


    print('result_trigger',result_trigger)
    pd.DataFrame(result_trigger).to_csv("score.csv")

    # obtains_data = pd.DataFrame({
    # 'Time': time_values,
    # 'Mouse': mouse_values,
    # 'Reaction': reaction_values,
    # 'Left eye': left_eye_values,
    # 'Right eye': right_eye_values,
    # 'Keyboard': keyboard_values,
    # 'Heart Rate': heart_rate_values,
    # 'Engagement': result_trigger
    # })
    
    # #print(obtains_data)
    # file_name = 'ObtainsData.xlsx'
            
    # #saving the excel
    # obtains_data.to_excel(file_name)
    
    # pd.DataFrame(result_trigger).to_csv("score.csv")
    # print('DataFrame is written to Excel File successfully.')
    # When everything is done, release the capture
    cam.release()
    cv2.destroyAllWindows()
    keyboard_listener.stop()
    
    

    create_real_time_dashboard( array_distg[-100:], mouse_interaction_distancesk[-100:],  heart_mean_diffsk[-100:], array_distg[-100:],'update')
    #print(mouse_values)

    score_sum = engagement_count_sk + bored_count_sk + frustrated_count_sk
    
    engagement_count_pr = (engagement_count_sk * 100/score_sum)
    bored_count_pr     =  (bored_count_sk * 100/score_sum)
    frustrated_count_pr = (frustrated_count_sk * 100/score_sum)

    print('Engagement Score',engagement_count_pr)
    print('Boredom Score',(bored_count_sk * 100/score_sum))
    print('Frustration Score',(frustrated_count_sk * 100/score_sum))
    
    
    # Define the message using f-string
    message = f"Engagement Score: {(engagement_count_sk * 100/score_sum)}%\n" \
            f"Boredom Score: {(bored_count_sk * 100/score_sum)}%\n" \
            f"Frustration Score: {(frustrated_count_sk * 100/score_sum)}%" 
    
    learningState_data = 1
    if max([engagement_count_sk,bored_count_sk,frustrated_count_sk])==bored_count_sk:
        learningState_data = 2
    if max([engagement_count_sk,bored_count_sk,frustrated_count_sk])==frustrated_count_sk:
        learningState_data = 3
    # # Define the message using f-string
    # message = f"Left Eye Interaction: {left_eye_interaction}\n" \
    #         f"Right Eye Interaction: {right_eye_interaction}\n" \
    #         f"Emotion Median Value: {emotion_median_value}\n" \
    #         f"Mouse Interaction Distance: {mouse_interaction_distance}\n" \
    #         f"Keyboard Interaction Score: {keyboard_interaction_score}\n" \
    #         f"Heart Mean Difference: {heart_mean_diff}\n" \
    #         f"Result: {classification}"

    # Define the title
    title = "Engagement Analysis"
    # Display the message box
    ctypes.windll.user32.MessageBoxW(0, message, title, 1)

        # Function to change the color of a ball based on the percentage
    def change_color_by_percentage(ball_item, percentage):
        if percentage < 33:
            canvas.itemconfig(ball_item, fill="blue")
        elif 33 <= percentage < 66:
            canvas.itemconfig(ball_item, fill="yellow")
        else:
            canvas.itemconfig(ball_item, fill="green")

    # Function to show color balls with gaps and labels
    def showColorBalls(engagement_count, boredom_count, frustration_count):
        total = engagement_count + boredom_count + frustration_count
        if total == 0:
            return

        # Calculate percentages
        engagement_percentage = (engagement_count / total) * 100
        boredom_percentage = (boredom_count / total) * 100
        frustration_percentage = (frustration_count / total) * 100

        # Update the colors of the balls based on percentages
        change_color_by_percentage(engagement_ball, engagement_percentage)
        change_color_by_percentage(boredom_ball, boredom_percentage)
        change_color_by_percentage(frustration_ball, frustration_percentage)

        # Update the labels under the balls
        engagement_label.config(text=f"Engagement: {engagement_count}")
        boredom_label.config(text=f"Boredom: {boredom_count}")
        frustration_label.config(text=f"Frustration: {frustration_count}")

    # Create the tkinter window
    window = tk.Tk()
    window.title("Color Balls with Labels")

    # Create a canvas to draw the balls
    canvas = tk.Canvas(window, width=400, height=200)
    canvas.pack()

    # Create balls for engagement, boredom, and frustration with gaps
    ball_radius = 30
    ball_spacing = 30
    engagement_ball = canvas.create_oval(50, 50, 50 + 2 * ball_radius, 50 + 2 * ball_radius, fill="gray")
    boredom_ball = canvas.create_oval(50 + 2 * ball_radius + ball_spacing, 50, 50 + 4 * ball_radius + ball_spacing, 50 + 2 * ball_radius, fill="gray")
    frustration_ball = canvas.create_oval(50 + 4 * ball_radius + 2 * ball_spacing, 50, 50 + 6 * ball_radius + 2 * ball_spacing, 50 + 2 * ball_radius, fill="gray")

    # Create labels under the balls
    engagement_label = tk.Label(window, text="Engagement: 0")
    engagement_label.place(x=50 + ball_radius - 40, y=110)
    boredom_label = tk.Label(window, text="Boredom: 0")
    boredom_label.place(x=50 + 3 * ball_radius + ball_spacing - 40, y=110)
    frustration_label = tk.Label(window, text="Frustration: 0")
    frustration_label.place(x=50 + 5 * ball_radius + 2 * ball_spacing - 40, y=110)

    showColorBalls(engagement_count=engagement_count_pr,boredom_count=bored_count_pr,frustration_count=frustrated_count_pr)
        # Start the tkinter main loop
    window.mainloop()
 
    #API CALLING POST METHOD
    #Define the URL where you want to send the JSON data
    url = 'https://softqnrapi20230628122139.azurewebsites.net/api/Students'  # Replace with the actual API endpoint URL

    # Create a Python dictionary with the data you want to send
    data_to_send = {
                                    "classId" : "1", # Classroom ID
                                    "studentId" : "234", # Student ID
                                    "studentName" : "fahad", # Student Name
                                    "behaviouralState" : "1", # Status of the student presented in 3 values: 1 green, 2 orange, 3 red
                                    "learningState" : learningState_data, # Status of the student presented in 3 values: 1 blue, 2 grey, 3 red
                                    "heartRate" : "75",  # no of heartbeats per minute
                                    "emotionalState" : "Neutral", # Anger, Fear, Happiness, Sadness, Disgust, Surprise
                    }


    # Convert the Python dictionary to a JSON string
    json_data = json.dumps(data_to_send)

    # Set the headers to indicate that you are sending JSON data
    headers = {
        'Content-Type': 'application/json'
    }

    # Send a POST request with the JSON data
    response = requests.post(url, data=json_data, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        print('Data sent successfully.')
    else:
        print('Failed to send data. Status code:', response.status_code)



    