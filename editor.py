import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple
from moviepy import VideoFileClip, CompositeVideoClip

@dataclass
class AudioAnalysisConfig:
    sample_rate: int = 44100
    chunk_duration: float = 0.1  # seconds
    spike_threshold_percentile: float = 95
    highlight_buffer: int = 5  # seconds

class HighlightDetector:
    def __init__(self, video_path: str, config: AudioAnalysisConfig = None):
        self.video_path = video_path
        self.config = config or AudioAnalysisConfig()
        self.clip = None
        self.final_video = None
        self.rms_values = []
        self.merged_intervals = []

    def load_video(self):
        try:
            self.clip = VideoFileClip(self.video_path)
            self.final_video = CompositeVideoClip([self.clip])
        except Exception as e:
            raise RuntimeError(f"Failed to load video: {e}")

    def analyze_audio(self):
        audio = self.final_video.audio
        audio_array = audio.to_soundarray(fps=self.config.sample_rate)
        chunk_size = int(self.config.sample_rate * self.config.chunk_duration)
        
        for start_idx in range(0, len(audio_array), chunk_size):
            chunk = audio_array[start_idx:min(start_idx + chunk_size, len(audio_array))]
            if len(chunk) == 0:
                continue
            
            chunk_mono = np.mean(chunk, axis=1)
            rms = np.sqrt(np.mean(chunk_mono**2))
            self.rms_values.append(rms)

    def detect_spikes(self) -> List[float]:
        threshold = np.percentile(self.rms_values, self.config.spike_threshold_percentile)
        spikes = [i for i, val in enumerate(self.rms_values) if val > threshold]
        spike_times = [i * self.config.chunk_duration for i in spikes]
        return list(dict.fromkeys(np.round(spike_times)))

    def create_intervals(self, spike_times: List[float]) -> List[Tuple[float, float]]:
        intervals = []
        for spike_time in sorted(spike_times):
            start = max(0, spike_time - self.config.highlight_buffer)
            end = min(spike_time + self.config.highlight_buffer, self.clip.duration)
            intervals.append((start, end))
        return intervals

    def merge_intervals(self, intervals: List[Tuple[float, float]]):
        if not intervals:
            return
            
        intervals.sort(key=lambda x: x[0])
        current_start, current_end = intervals[0]
        
        for next_start, next_end in intervals[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                self.merged_intervals.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        self.merged_intervals.append((current_start, current_end))

    def export_highlights(self, output_dir: str, verbose: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        try:
            for idx, (start, end) in enumerate(self.merged_intervals, 1):
                if verbose:
                    print(f"Processing highlight {idx}/{len(self.merged_intervals)}")
                
                highlight = self.final_video.subclipped(start, end)
                clip_path = os.path.join(output_dir, f"highlight-{idx}.mp4")
                highlight.write_videofile(clip_path)
                highlight.close()
        finally:
            self.cleanup()

    def cleanup(self):
        if self.final_video:
            self.final_video.close()
        if self.clip:
            self.clip.close()

    def process(self, output_dir: str, verbose: bool = False):
        """Main processing pipeline"""
        self.load_video()
        self.analyze_audio()
        spike_times = self.detect_spikes()
        intervals = self.create_intervals(spike_times)
        self.merge_intervals(intervals)
        self.export_highlights(output_dir, verbose)
