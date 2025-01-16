import os
import cv2
from moviepy import VideoFileClip
from typing import Optional


class VideoFrameExtractor:
    def __init__(self, video_path: str, output_dir: str, fps: int = 10):
        """
        Initialize the frame extractor with video path and output settings.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory where frames will be saved
            fps (int): Number of frames to extract per second
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        self.video: Optional[VideoFileClip] = None
        
    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_video(self) -> None:
        """Load the video file."""
        self.video = VideoFileClip(self.video_path)
        
    def _save_frame(self, frame, frame_number: int) -> None:
        """Save an individual frame to the output directory."""
        frame_path = os.path.join(self.output_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    def extract(self) -> None:
        """Extract frames from the video."""
        try:
            self._setup_output_directory()
            self._load_video()
            
            total_frames = int(self.video.duration * self.fps)
            
            for frame_number in range(total_frames):
                frame = self.video.get_frame(frame_number / self.fps)
                self._save_frame(frame, frame_number)
                
            print(f"Successfully extracted {total_frames} frames")
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            
        finally:
            if self.video:
                self.video.close()
                
    @property
    def frame_count(self) -> int:
        """Get the total number of frames that will be extracted."""
        if self.video:
            return int(self.video.duration * self.fps)
        return 0


# Example usage
if __name__ == "__main__":
    extractor = VideoFrameExtractor("highlight-1.mp4", "frames/")
    extractor.extract()

