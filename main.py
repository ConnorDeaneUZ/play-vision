from preprocessing.video_editor import HighlightDetector
from preprocessing.frame_extractor import VideoFrameExtractor
from preprocessing.feature_extractor import FeatureExtractor
import os

def process_clips(clips_dir, frames_dir, features_dir):

    # # # Initialize the highlight detector with the source video
    # highlight_detector = HighlightDetector('southhampton.mp4')
    # # Run the detection process
    # highlight_detector.process(output_folder=features_dir, verbose=True)

    
    """Process clips from a directory and extract their features"""
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Create a frames directory for each clip
    for clip_file in os.listdir(clips_dir):
        if clip_file.endswith(".mp4"):
            clip_path = os.path.join(clips_dir, clip_file)
            clip_name = os.path.splitext(clip_file)[0]
            
            # Create a unique folder for each clip's frames
            output_folder = os.path.join(frames_dir, clip_name)
            
            frames_extractor = VideoFrameExtractor(clip_path)
            frames_extractor.output_folder = output_folder
            frames_extractor.extract()

            # Extract features for this specific clip
            feature_extractor = FeatureExtractor()
            feature_path = os.path.join(features_dir, f"{clip_name}.pt")
            feature_extractor.extract_from_directory(
                frame_dir=output_folder,
                file_extension=".jpg",
                save_path=feature_path
            )

def main():
    try:
        # Process goal clips
        process_clips(
            clips_dir="training/clips/goals",
            frames_dir="frames/goals",
            features_dir="features/goal"
        )
        
        # # Process no-goal clips
        process_clips(
            clips_dir="training/clips/no_goals",
            frames_dir="frames/no_goals",
            features_dir="features/no_goal"
        )

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
