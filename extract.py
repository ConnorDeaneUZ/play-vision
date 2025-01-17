import os
import cv2
from moviepy import VideoFileClip
from typing import Optional


class VideoFrameExtractor:
    def __init__(self, video_path: str, output_folder: str, frames_per_second: int = 10):
        """
        Sets up the video frame extractor with basic settings.
        
        Args:
            video_path (str): Where your video file is located (e.g., "my_video.mp4")
            output_folder (str): Where you want to save the extracted frames
            frames_per_second (int): How many frames to extract each second (default: 10)
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.frames_per_second = frames_per_second
        self.loaded_video: Optional[VideoFileClip] = None
        
    def _create_output_folder(self) -> None:
        """Creates a folder to store the extracted frames if it doesn't exist yet."""
        os.makedirs(self.output_folder, exist_ok=True)
        
    def _open_video(self) -> None:
        """Opens the video file so we can read from it."""
        self.loaded_video = VideoFileClip(self.video_path)
        
    def _save_single_frame(self, image_data, frame_number: int) -> None:
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
        
    def extract(self) -> None:
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
                
    @property
    def frame_count(self) -> int:
        """
        Calculates how many frames will be extracted from the video.
        Returns 0 if no video is loaded yet.
        """
        if self.loaded_video:
            return int(self.loaded_video.duration * self.frames_per_second)
        return 0



