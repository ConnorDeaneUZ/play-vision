import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple
from moviepy import VideoFileClip, CompositeVideoClip

@dataclass
class AudioAnalysisConfig:
    # Audio settings for processing
    sample_rate: int = 44100  # Number of audio samples per second (CD quality)
    chunk_duration: float = 0.1  # How long each audio segment is (in seconds)
    spike_threshold_percentile: float = 95  # Threshold to detect exciting moments
    highlight_buffer: int = 5  # Extra seconds to include before and after exciting moments

class HighlightDetector:
    def __init__(self, video_path: str, config: AudioAnalysisConfig = None):
        """Initialize the highlight detector with a video file"""
        self.video_path = video_path
        self.config = config or AudioAnalysisConfig()
        self.video_clip = None  # Original video
        self.processed_video = None  # Video after processing
        self.audio_intensity_values = []  # Store audio loudness measurements
        self.highlight_segments = []  # Store time ranges for highlights

    def load_video(self):
        """Load the video file for processing"""
        try:
            self.video_clip = VideoFileClip(self.video_path)
            self.processed_video = CompositeVideoClip([self.video_clip])
        except Exception as e:
            raise RuntimeError(f"Could not open video file: {e}")

    def analyze_audio(self):
        """Break down the audio into chunks and measure their intensity"""
        audio = self.processed_video.audio
        raw_audio = audio.to_soundarray(fps=self.config.sample_rate)
        samples_per_chunk = int(self.config.sample_rate * self.config.chunk_duration)
        
        # Process audio in small chunks
        for chunk_start in range(0, len(raw_audio), samples_per_chunk):
            audio_chunk = raw_audio[chunk_start:min(chunk_start + samples_per_chunk, len(raw_audio))]
            if len(audio_chunk) == 0:
                continue
            
            # Convert stereo to mono by averaging channels
            mono_audio = np.mean(audio_chunk, axis=1)
            # Calculate root mean square (loudness) of the chunk
            loudness = np.sqrt(np.mean(mono_audio**2))
            self.audio_intensity_values.append(loudness)

    def detect_spikes(self) -> List[float]:
        """Find moments where audio is louder than usual"""
        # Calculate threshold based on percentile of loudness values
        loudness_threshold = np.percentile(self.audio_intensity_values, self.config.spike_threshold_percentile)
        # Find times where audio exceeds threshold
        loud_moments = [i for i, loudness in enumerate(self.audio_intensity_values) if loudness > loudness_threshold]
        # Convert chunk indices to actual timestamps
        timestamp_seconds = [i * self.config.chunk_duration for i in loud_moments]
        # Remove duplicates and round to nearest second
        return list(dict.fromkeys(np.round(timestamp_seconds)))

    def create_intervals(self, exciting_moments: List[float]) -> List[Tuple[float, float]]:
        """Create time ranges around exciting moments"""
        time_ranges = []
        for moment in sorted(exciting_moments):
            # Add buffer time before and after each moment
            start_time = max(0, moment - self.config.highlight_buffer)
            end_time = min(moment + self.config.highlight_buffer, self.video_clip.duration)
            time_ranges.append((start_time, end_time))
        return time_ranges

    def merge_intervals(self, time_ranges: List[Tuple[float, float]]):
        """Combine overlapping time ranges into longer segments"""
        if not time_ranges:
            return
            
        time_ranges.sort(key=lambda x: x[0])  # Sort by start time
        current_start, current_end = time_ranges[0]
        
        for next_start, next_end in time_ranges[1:]:
            if next_start <= current_end:  # If ranges overlap
                current_end = max(current_end, next_end)  # Extend current range
            else:
                self.highlight_segments.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        self.highlight_segments.append((current_start, current_end))

    def export_highlights(self, output_folder: str, verbose: bool = False):
        """Save highlight clips as separate video files"""
        os.makedirs(output_folder, exist_ok=True)
        try:
            for clip_number, (start_time, end_time) in enumerate(self.highlight_segments, 1):
                if verbose:
                    print(f"Saving highlight clip {clip_number} of {len(self.highlight_segments)}")
                
                highlight_clip = self.processed_video.subclipped(start_time, end_time)
                output_path = os.path.join(output_folder, f"highlight-{clip_number}.mp4")
                highlight_clip.write_videofile(output_path)
                highlight_clip.close()
        finally:
            self.cleanup()

    def cleanup(self):
        """Release video resources"""
        if self.processed_video:
            self.processed_video.close()
        if self.video_clip:
            self.video_clip.close()

    def process(self, output_folder: str, verbose: bool = False):
        """Main processing pipeline to extract video highlights"""
        self.load_video()
        self.analyze_audio()
        exciting_moments = self.detect_spikes()
        time_ranges = self.create_intervals(exciting_moments)
        self.merge_intervals(time_ranges)
        self.export_highlights(output_folder, verbose)
