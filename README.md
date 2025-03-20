# Sentimental_analysis
# Real-time Emotion Detection with DeepFace and OpenCV

This project uses OpenCV and DeepFace to perform real-time emotion detection from a webcam feed. It displays the detected emotion, a confidence score, and a dynamic bar chart visualizing the probabilities of all recognized emotions.

## Features

*   **Real-time Emotion Recognition:** Analyzes video frames from your webcam to identify facial expressions.
*   **Multiple Emotions:** Detects seven emotions: angry, disgust, fear, happy, sad, surprise, and neutral.
*   **Confidence Threshold:**  Filters out low-confidence detections to improve accuracy.  A configurable confidence threshold (`CONFIDENCE_THRESHOLD`) determines the minimum confidence level for an emotion to be displayed.
*   **Emotion Chart:**  Displays a real-time bar chart showing the confidence levels for all emotions.
*   **Face Padding:** Adds padding around detected faces to ensure the entire face is included in the analysis.  This helps prevent cropping important facial features.
*   **Error Handling:** Includes robust error handling to gracefully manage situations where face detection or emotion analysis fails.  This prevents the program from crashing and provides informative messages.
*   **Adjustable Parameters:**  Allows easy configuration of detection parameters, appearance settings, and window size through clearly defined constants.
*   **Clear Output:**  Draws bounding boxes around detected faces, labels them with the dominant emotion (or "Uncertain" / "No Emotion" if necessary), and updates the emotion chart.
* **Window Resizing**: Resizes the video to the specified `WINDOW_WIDTH` and `WINDOW_HEIGHT` if the captured video's dimensions don't match, maintaining consistent display.

## Prerequisites

1.  **Python:**  Ensure you have Python 3.6 or higher installed.  You can download it from [python.org](https://www.python.org/).

2.  **pip:**  The Python package installer.  It usually comes bundled with Python installations.  You can check if you have it by running `pip --version` in your terminal/command prompt.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/<your_username>/<your_repository_name>.git  # Replace with your repo URL
    cd <your_repository_name>
    ```
    *Replace `<your_username>` and `<your_repository_name>` with your actual GitHub username and repository name.*

2.  **Install Dependencies:**

    ```bash
    pip install opencv-python deepface numpy
    ```
    This command installs the required libraries:
    *   `opencv-python`:  For computer vision tasks (webcam access, image processing, drawing).
    *   `deepface`:  For face recognition and emotion analysis.  This library will automatically download necessary models.
    *   `numpy`:  For numerical operations (used internally by OpenCV and DeepFace).

## Running the Emotion Detection

1.  **Navigate to the project directory:**

    ```bash
    cd <your_repository_name>  # If you're not already there
    ```

2.  **Run the script:**

    ```bash
    python emotion_detection.py
    ```
    *Replace `emotion_detection.py` with the actual name of your Python script if it's different.*

3.  **Webcam Access:**  The script will attempt to access your default webcam (usually camera index 0).  You might be prompted by your operating system to grant permission to the application to access your webcam.

4.  **Quit:**  Press the 'q' key to terminate the program and close the webcam window.

## Configuration

You can customize the behavior of the emotion detection by modifying the constants at the beginning of the `emotion_detection.py` script.  Here's a breakdown:

*   **`CONFIDENCE_THRESHOLD`:**  (Default: 10)  Minimum confidence (in percentage) for an emotion to be displayed.
*   **`DETECTOR_BACKEND`:** (Default: 'opencv')  The face detector backend to use. DeepFace supports several backends ('opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface').  'opencv' is generally a good default choice for speed and compatibility.
*   **`SCALE_FACTOR`:**  (Default: 1.3)  Parameter for the face detection cascade classifier.  Smaller values increase detection accuracy but slow down processing.
*   **`MIN_NEIGHBORS`:**  (Default: 5)  Parameter for the face detection cascade classifier.  Higher values reduce false positives but might miss some faces.
*   **`MIN_SIZE`:**  (Default: (40, 40))  Minimum face size (width, height) in pixels to be detected.
*   **`PADDING_X_PERCENT`:** (Default: 0.15)  Percentage of face width to add as padding on the left and right sides.
*   **`PADDING_Y_PERCENT`:** (Default: 0.25) Percentage of face height to add as padding on the top and bottom sides.
*   **`RECTANGLE_COLOR`:**  (Default: (0, 255, 0) - Green)  Color of the bounding box around detected faces.
*   **`TEXT_COLOR`:** (Default: (255, 255, 255) - White) Color of the emotion label text.
*   **`UNCERTAIN_COLOR`:** (Default: (0, 0, 255)- Red) Color to be used for the bounding box and text when the confidence level is below the `CONFIDENCE_THRESHOLD` or when no emotion is detected.
*   **`FONT`:** (Default: `cv2.FONT_HERSHEY_SIMPLEX`) Font for the text labels.
*   **`FONT_SCALE`:** (Default: 0.6) Font size.
*   **`FONT_THICKNESS`:** (Default: 1) Thickness of the text.
*   **`BAR_WIDTH`:**  (Default: 20)  Width of each bar in the emotion chart.
*   **`BAR_SPACING`:** (Default: 5)  Spacing between bars in the emotion chart.
*   **`EMOTIONS`:**  List of emotions to be detected.  *Do not modify this unless you know what you're doing.*
*   **`BAR_COLORS`:**  Dictionary mapping emotion names to their corresponding bar colors in the chart.
*  **`WINDOW_WIDTH`:** (Default: 1280) Initial width of the display window.
*  **`WINDOW_HEIGHT`:** (Default: 720) Initial height of the display window.

## Troubleshooting

*   **"Cannot open webcam" error:**
    *   Make sure your webcam is properly connected and recognized by your system.
    *   Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`, `cv2.VideoCapture(2)`, etc., if you have multiple cameras.
    *  Ensure no other application is currently using the webcam.
*   **"Detected face ROI is empty" warning:**
    *   This usually means the detected face region is too small or invalid. Try adjusting `SCALE_FACTOR`, `MIN_NEIGHBORS`, and `MIN_SIZE` in the configuration.  Make sure your face is well-lit and clearly visible to the camera.
* **Slow Performance:**
    *   Reduce the `WINDOW_WIDTH` and `WINDOW_HEIGHT`
    *   Try a different `DETECTOR_BACKEND` (e.g., 'ssd' might be faster on some systems).
    *   Increase `SCALE_FACTOR` (trade-off: may decrease detection accuracy).
    *   Increase `MIN_SIZE` (trade-off: may miss smaller faces).

*   **DeepFace errors:**
    *   Make sure you have an internet connection the first time you run the script, as DeepFace will download necessary model files.
    * If errors persist, consult the [DeepFace documentation](https://github.com/serengil/deepface) for more specific troubleshooting.

*   **General Errors:**  The script includes `try...except` blocks to catch and print any unexpected errors.  If you encounter an error, check the console output for details.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests.  If you find any bugs or have suggestions for improvements, please open an issue on the GitHub repository.
