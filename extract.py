import os
import cv2
from moviepy import VideoFileClip
from typing import Optional


class VideoExtractorConfig:
    frames_per_second: int = 10
    output_folder: str = "frames/"

class VideoFrameExtractor:
    def __init__(self, video_path: str):
        """Sets up the video frame extractor with basic settings."""
        self.video_path = video_path
        self.output_folder = VideoExtractorConfig.output_folder
        self.frames_per_second = VideoExtractorConfig.frames_per_second
        self.loaded_video: VideoFileClip | None = None
        
    def _create_output_folder(self):
        """Creates a folder to store the extracted frames if it doesn't exist yet."""
        os.makedirs(self.output_folder, exist_ok=True)
        
    def _open_video(self):
        """Opens the video file so we can read from it."""
        self.loaded_video = VideoFileClip(self.video_path)
        
    def _save_single_frame(self, image_data, frame_number: int):
        """
        Saves one frame as an image file.
        
        Args:
            image_data: The picture data to save
            frame_number: Which frame number this is (used in filename)
        """
        # Create the filename for this frame (e.g., "frame_0.jpg")
        image_path = os.path.join(self.output_folder, f"frame_{frame_number}.jpg")
        # Convert the image colors from RGB to BGR (which is what OpenCV expects)
        image_data_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        # Save the image to disk
        cv2.imwrite(image_path, image_data_bgr)
        
    def extract(self):
        """
        Main function that extracts all frames from the video.
        This will create numbered image files in your output folder.
        """
        try:
            # First, set everything up
            self._create_output_folder()
            self._open_video()
            
            # Calculate how many frames we'll extract in total
            total_frames_to_extract = int(self.loaded_video.duration * self.frames_per_second)
            
            # Extract each frame one by one
            for current_frame_number in range(total_frames_to_extract):
                # Calculate the exact time in the video for this frame
                time_in_seconds = current_frame_number / self.frames_per_second
                # Get the image data for this moment in the video
                frame_image = self.loaded_video.get_frame(time_in_seconds)
                # Save this frame as an image file
                self._save_single_frame(frame_image, current_frame_number)
                
            print(f"Success! Extracted {total_frames_to_extract} frames from the video")
            
        except Exception as error:
            print(f"Error while extracting frames: {str(error)}")
            
        finally:
            # Always close the video file when we're done
            if self.loaded_video:
                self.loaded_video.close()
