import os
import cv2
from moviepy import VideoFileClip


def extract_frames(video_path, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)

    count = 0
    for t in range(0, int(video.duration * fps)):
        frame = video.get_frame(t / fps)
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        count += 1
    print("Frame extraction successfull")


extract_frames("highlight-1.mp4", "frames/")

