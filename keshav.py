import cv2
import numpy as np
import pygame

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Function to detect fire
def detect_fire(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the fire color in HSV
    lower_fire = np.array([5, 150, 150])  # Lower bound for fire color (orange/red)
    upper_fire = np.array([15, 255, 255])  # Upper bound for fire color (orange/red)

    # Create a mask for detecting fire
    fire_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)

    # Perform bitwise AND to get the fire regions
    fire_detection = cv2.bitwise_and(frame, frame, mask=fire_mask)

    # Count non-zero pixels in the fire mask to check if fire is detected
    fire_pixels = cv2.countNonZero(fire_mask)

    return fire_detection, fire_mask, fire_pixels

# Load the sound file
sound_file = 'f.mp3'  # Specify the path to your alert sound file
try:
    pygame.mixer.music.load(sound_file)
except pygame.error as e:
    print(f"Error loading sound file: {e}")
    exit()

# Initialize webcam video capture (can be replaced with a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Call the fire detection function
    fire_detection, fire_mask, fire_pixels = detect_fire(frame)

    # If fire pixels are detected, play a sound alert
    if fire_pixels > 1000:  # Adjust this threshold based on your needs
        if not pygame.mixer.music.get_busy():  # Ensure the sound doesn't overlap
            pygame.mixer.music.play()

    # Display the original frame, the fire detection, and the fire mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Fire Detection', fire_detection)
    cv2.imshow('Fire Mask', fire_mask)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Stop any playing sound
pygame.mixer.music.stop()
pygame.mixer.quit()
