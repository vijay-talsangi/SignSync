# SignSync

**SignSync** is a Python-based application that utilizes machine learning, computer vision, and natural language processing to recognize sign language gestures and convert them to text or speech. It employs various libraries like Flask, OpenCV, and MediaPipe to provide a real-time sign language recognition system.

---

## Features

- **Sign Language Recognition**: Detects hand gestures and translates them into text.
- **Text-to-Speech Conversion**: Converts recognized sign language text into speech using the Google Text-to-Speech (gTTS) library.
- **Real-Time Feedback**: Provides real-time feedback on the detected sign language through both text and voice.
- **Web Interface**: Allows interaction via a web interface powered by Flask and WebSockets.

---

## Requirements

### Libraries & Dependencies
The project requires several Python libraries for its functionality. You can easily install them using `requirements.txt`. Here's a list of the required libraries:

- Flask
- Flask-SocketIO
- OpenCV
- pandas
- scikit-learn
- MediaPipe
- Pygame
- gTTS (Google Text-to-Speech)

---

## Installation

To set up **SignSync** on your machine, follow these steps:

1. **Clone the repository**:

   ```
   git clone https://github.com/yourusername/SignSync.git
   cd SignSync
   ```
2. **Set Up a Virtual Environment (Recommended)**:
   ```
   python3 -m venv .venv
   ```
   - For Windows:
     ```
     venv\Scripts\activate
     ```

   - For Mac/Linux:
     ```
     source venv/bin/activate
     ```
3. **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```
4. **Run the Flask Web Server**:
    ```
     python app.py
    ```
## Access Your Web App on:
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)


    