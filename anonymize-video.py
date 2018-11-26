#! /usr/bin/env python3

# Copyright (c) 2018 Florent Revest
# File distributed under the terms of the MIT license. (See COPYING.MIT)

import sys
import cv2

from face_tools import dream_faces, find_faces, overwrite_face

# If we don't have the right number of parameters, print usage and quit
if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " input_video output_video")
    sys.exit(1)

# Open the input and output video files (with OpenCV)
print("Opening videos...")
input_video = cv2.VideoCapture(sys.argv[1])
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter(sys.argv[2], 0, 25.0, (width, height))

# Use the Progressive GANs model to "dream" celebrities (with TensorFlow)
dreamed_celebrities_nb = 10
print("Dreaming " + str(dreamed_celebrities_nb) + " fake celebrities...")
dreamed_imgs = dream_faces(dreamed_celebrities_nb)

# Localize each celebrity's face and landmarks (with dlib)
print("Localizing their faces...")
dreamed_faces = []
for dreamed_img in dreamed_imgs:
    faces = find_faces(dreamed_img)

    if len(faces) != 1:
        print(str(len(faces)) + " faces found in a dreamed image. Aborting.")
        sys.exit(2)

    dreamed_faces.append(faces[0])

print("Processing video...")
# Iterate over every frame of the input video 
total_frames_nb = str(int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)))
while True:
    ret, video_frame = input_video.read()
    if ret == 0:
        break
    curr_frame_nb = str(int(input_video.get(cv2.CAP_PROP_POS_FRAMES)))
    print("[" + curr_frame_nb + "/" + total_frames_nb + "]", end='', flush=True)

    # Localize faces in the frame
    video_faces = find_faces(video_frame)
    detected_faces_nb = len(video_faces)
    print(" Replacing " + str(detected_faces_nb) + " faces...")
    if detected_faces_nb > dreamed_celebrities_nb:
        print(str(len(faces)) + " found (> " + str(dreamed_celebrities_nb) +
                " in an input frame. Aborting.")
        sys.exit(3)

    # Replace each detected face with a dreamed celebrity
    for i in range(len(video_faces)):
        overwrite_face(dreamed_faces[i], video_faces[i], video_frame)

    # Append the modified frame to the output
    output_video.write(video_frame)

# Close the input and output video files
input_video.release()
output_video.release()

print("Anonymized video saved to: " + sys.argv[2])
