import cv2
from deepface import DeepFace
import numpy as np

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Configuration ---
CONFIDENCE_THRESHOLD = 10
DETECTOR_BACKEND = 'opencv'
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_SIZE = (40, 40)
PADDING_X_PERCENT = 0.15
PADDING_Y_PERCENT = 0.25
RECTANGLE_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
UNCERTAIN_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6  # Reduced font scale
FONT_THICKNESS = 1  # Reduced font thickness
BAR_WIDTH = 20  # Further reduced width
BAR_SPACING = 5   # Further reduced spacing
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
BAR_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (0, 255, 255),
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (255, 0, 255),
    'neutral': (128, 128, 128)
}
# --- Window Size ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Helper function to draw the emotion chart ---
def draw_emotion_chart(frame, emotion_data, chart_x, chart_y, bar_width, bar_spacing, max_bar_height):
    """Draws a vertical bar chart with improved spacing and smaller text."""
    total_chart_width = len(EMOTIONS) * (bar_width + bar_spacing) - bar_spacing
    chart_x = frame.shape[1] - total_chart_width - 40

    for i, emotion in enumerate(EMOTIONS):
        confidence = emotion_data.get(emotion, 0)
        bar_height = int(max_bar_height * confidence / 100)
        bar_x = chart_x + (bar_width + bar_spacing) * i
        bar_y = chart_y - bar_height
        color = BAR_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, chart_y), color, -1)

        # Smaller text and adjusted positioning
        label = f"{emotion}: {confidence:.0f}%"
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]  # Use config values
        text_x = bar_x + bar_width // 2 - text_size[0] // 2
        text_y = chart_y + 14  # Slightly closer to the bar
        cv2.putText(frame, label, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

# --- Main Program ---
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set initial window size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_bar_height = int(frame_height * 0.2)
    chart_y = frame_height - 50


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame, exiting...")
            break
        frame = cv2.flip(frame, 1)

        if frame.shape[1] != WINDOW_WIDTH or frame.shape[0] != WINDOW_HEIGHT:
           frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE)

        for (x, y, w, h) in faces:
            padding_x = int(w * PADDING_X_PERCENT)
            padding_y = int(h * PADDING_Y_PERCENT)
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(frame.shape[1], x + w + padding_x)
            y2 = min(frame.shape[0], y + h + padding_y)
            face_roi = frame[y1:y2, x1:x2]

            try:
                if face_roi.size == 0:
                    print("Detected face ROI is empty, skipping analysis.")
                    continue

                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True, detector_backend=DETECTOR_BACKEND)

                if result and result[0].get('emotion'):
                    emotion_data = result[0]['emotion']
                    dominant_emotion = max(emotion_data, key=emotion_data.get)
                    confidence = emotion_data[dominant_emotion]

                    if confidence > CONFIDENCE_THRESHOLD:
                        label = f"{dominant_emotion} ({confidence:.0f}%)"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, FONT_THICKNESS)
                        cv2.putText(frame, label, (x, y - 10), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), UNCERTAIN_COLOR, FONT_THICKNESS)
                        cv2.putText(frame, "Uncertain", (x, y - 10), FONT, FONT_SCALE, UNCERTAIN_COLOR, FONT_THICKNESS)

                    draw_emotion_chart(frame, emotion_data, 0, chart_y, BAR_WIDTH, BAR_SPACING, max_bar_height)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), UNCERTAIN_COLOR, FONT_THICKNESS)
                    cv2.putText(frame, "No Emotion", (x, y - 10), FONT, FONT_SCALE, UNCERTAIN_COLOR, FONT_THICKNESS)

            except Exception as e:
                print(f"Error during DeepFace analysis: {e}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), UNCERTAIN_COLOR, FONT_THICKNESS)
                cv2.putText(frame, "Unknown", (x, y - 10), FONT, FONT_SCALE, UNCERTAIN_COLOR, FONT_THICKNESS)
                continue

        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()