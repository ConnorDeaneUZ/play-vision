from preprocessing.video_editor import HighlightDetector
from preprocessing.frame_extractor import VideoFrameExtractor
from preprocessing.feature_extractor import FeatureExtractor

def main():
    try:
        detector = HighlightDetector("liverpool-vs-united.mp4")
        detector.process("clips", verbose=False)
        
        # Only extract frames after highlights are processed successfully
        frames_extractor = VideoFrameExtractor("clips/highlight-1.mp4")
        frames_extractor.extract()

        # Extract features from frames after extraction
        feature_extractor = FeatureExtractor()
        feature_extractor.extract_from_directory(
            frame_dir="frames/", 
            file_extension=".jpg", 
            save_path="features/features.pt"
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
