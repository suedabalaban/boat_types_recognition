#import the necessary libraries
import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

#load the model
model = YOLO("boat_types.pt")

#open video file
cap = cv2.VideoCapture("source/ferry_frontview.mp4")
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

#Get resolution and FPS of input video
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#set frame skip variables
frame_skip = 2
frame_count = 0

#a list to keep processed frames
processed_frames = []

#process video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    #speed up the image acquisition process
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    #resize resolution of frame
    frame = cv2.resize(frame, (input_width, input_height))

    #process results using the model
    results = model(frame)

    #annotate the detected objects
    annotated_frame = results[0].plot()

    #Add the annotated frames to the processed_frames list
    processed_frames.append(annotated_frame)

    #Show the results
    cv2.imshow("Boat Type Detection", annotated_frame)

    #Get results every 1 ms, break the loop if the q key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Release resources and close windows
cap.release()
cv2.destroyAllWindows()

#Create video with MoviePy
clip = VideoFileClip("source/ferry_frontview.mp4")
annotated_clip = clip.fl_image(lambda img: processed_frames.pop(0) if processed_frames else img)
annotated_clip.write_videofile("results/output.mp4", codec="libx264", fps=fps)