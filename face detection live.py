import cv2
import os

# Path to the haarcascade XML file (adjust the path accordingly)
haar_cascade_path = "C:/Users/asus/path_to_haar_file/haarcascade_frontalface_default.xml"

# Check if the XML file exists at the specified location
if not os.path.isfile(haar_cascade_path):
    print(f"Error: {haar_cascade_path} not found! Please download the file and place it in the specified path.")
    exit()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not video_capture.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the classifier was loaded correctly
if faceCascade.empty():
    print("Error: Could not load face cascade classifier.")
    exit()

while(True):
    ret, frame = video_capture.read()  # Use video_capture instead of cap
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # If 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
