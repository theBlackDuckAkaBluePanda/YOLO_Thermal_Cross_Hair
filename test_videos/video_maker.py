"""
Overview: Compiles a sequence of images (e.g.  from a downloaded Kaggle dataset) into an MP4 video.
It auto-sorts the files, standardizes their resolution, and stitches them together at a set FPS.
"""

import cv2
import os

# settings
image_folder = ''
video_name = 'test_video.mp4'
fps = 1


# Retrieve and sort all JPG files from the specified folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

if not images:
    print("Error: No JPG files found in the specified folder!")
    exit()


# Read the first image to establish the video resolution
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

total_duration = len(images) / fps
print(f"Found {len(images)} images.")

#adding images to video
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)

    # Resize every frame to match the first image's dimensions to prevent writer errors
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()
