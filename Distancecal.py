import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Response
import numpy as np

# Function to calculate distance in centimeters
def calculate_distance(focal_length, known_width, per_width):
    return (known_width * focal_length) / per_width

# Initialize the camera and get focal length
def get_focal_length(frame_width, focal_length, known_width):
    return (frame_width * focal_length) / known_width

def get_camera_parameters():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None

    # Capture a frame to get image properties
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None, None

    # Get image dimensions
    height, width = frame.shape[:2]

    # Assume the known width (face width) occupies 1/3 of the screen width
    known_width = width / 3

    return cap, known_width

def capture_frame():
    # Get camera parameters
    cap, known_width = get_camera_parameters()
    if cap is None or known_width is None:
        return

    # Set threshold distances for alerts
    min_distance_cm = 35  # Minimum distance in centimeters
    max_distance_cm = 105  # Maximum distance in centimeters

    # Ideal distance between the user and the screen (adjust as needed)
    ideal_distance_cm = 65  # 40 centimeters

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Calculate aspect ratio to filter out non-face objects
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:  # Adjust these thresholds as needed
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Calculate distance to the face
                face_midpoint = x + w // 2
                
                # Calculate focal length using known width and distance
                focal_length = get_focal_length(frame.shape[1], known_width, w)
                
                # Calculate distance in pixels from the camera to the face
                distance_pixels = known_width * focal_length / w

                # Convert distance from pixels to centimeters
                distance_cm = distance_pixels / 10  # Assuming 1 cm = 10 pixels

                # Show distance on the frame
                cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Alert if distance is too close or too far
                if distance_cm < min_distance_cm:
                    cv2.putText(frame, "Move back", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif distance_cm > max_distance_cm:
                    cv2.putText(frame, "Move forward", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Encode frame as JPEG image
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera
    cap.release()

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div(style={'backgroundColor': '#f0f8ff', 'textAlign': 'center'}, children=[
    html.H1("Check your position", style={'color': 'darkblue', 'textAlign': 'center'}),
    html.Div(id='live-update-text'),
    html.Img(src="/live-feed")
])

# Define callback to serve live camera feed
@app.server.route('/live-feed')
def live_feed():
    return Response(capture_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run_server(debug=True)