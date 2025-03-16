import cv2
import numpy as np
from flask import Flask, render_template, Response, send_file
from flask_socketio import SocketIO
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time
import hand_detector2 as hdm
import pygame
from gtts import gTTS
import io
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the hand detector and the model
cap = cv2.VideoCapture(0)
detector = hdm.handDetector()
model = LogisticRegression(max_iter=200)

# Load the model (assuming the model has already been trained and saved)
data = pd.read_csv('hand_signals.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
X = data.drop('letter', axis=1)
y = data['letter']
model.fit(X, y)

def speech(text):
    '''
    Converts a piece of text into speech using pyGame and gTTS libraries

    Parameters:
    text (string): A string representing the text to be converted into speech

    Returns: None
    '''
    
    #Initializes the text
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    myobj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    filename = 'audio/output.mp3'
    myobj.save(filename)  # Save the generated speech as an MP3 file
    
    # Emit the filename to frontend
    socketio.emit('audio_file', 'output.mp3')

    # Load the BytesIO object as a sound
    # pygame.mixer.music.load(mp3_fp, 'mp3')
    # pygame.mixer.music.play()

    # # Keep the program running while the sound plays
    # while pygame.mixer.music.get_busy():
    #     pygame.time.Clock().tick(10)
def gen():
    """Generate video frames and send predictions to the frontend"""
    pygame.mixer.init()
    letters = [0]
    word = ''
    words = []
    start = time.time()
    end = time.time()
    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Detect hand landmarks
        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)
        
#Confidence threshold for Regressor model
        confidence_threshold = .7

        #Checks if hands aren't detected
        if not landmarks:

            if start == 0:  # If no hand has been detected previously
                start = time.time()  # Set the start time

            idle_timer = time.time() - start

            #Checks if inactivity timer has exceeded three seconds
            if idle_timer >= 3 and word != '':

                #Checks if there is a word to dictate
                if word[-1] != ' ':
                    
                    #Dictates the word and adds it to the words list
                    speech(word)
                    words.append(word)
                    word = ''

        # If hand is detected, predict the letter
        if landmarks and len(landmarks) == 1:
            lmlist = landmarks[0][1]
            #Stops inactivity timer 
            end = time.time()

            #Finds the highest and lowest points of each hand to draw the rectangle around the hand
            p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25, min(lmlist[x][2] for x in range(len(lmlist))) - 25)
            p2 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25, max(lmlist[x][2] for x in range(len(lmlist))) + 25)
            cv2.rectangle(img, p1, p2, (255,255,255), 3)
            location_vector = np.array([coord for lm in lmlist for coord in lm[1:3]]).reshape(1, -1)

            # Get model predictions
            probabilities = model.predict_proba(location_vector)
            max_prob = np.max(probabilities)
            confidence_threshold = 0.7

            if max_prob > confidence_threshold:
                predicted_letter = model.predict(location_vector)[0]
                if predicted_letter == letters[-1]:
                    letters.append(predicted_letter)
                else:
                    letters = [predicted_letter]
                cv2.putText(img, predicted_letter, (p1[0], p1[1] - 10), cv2.QT_FONT_NORMAL, 3, (255, 255, 255), 3)

                if len(letters) == 20:
                    word = word + letters[0]
                    letters = [0]
                    socketio.emit('word', word)  # Send word to frontend

        # Convert the image to JPEG and send it to the frontend
        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video frames to frontend"""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio/<filename>')
def audio(filename):
    return send_file(f'./audio/{filename}', mimetype='audio/mp3')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
