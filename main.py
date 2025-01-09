from moviepy import VideoFileClip, CompositeVideoClip
import numpy as np

# load video
clip = VideoFileClip("full-match.mp4")
final_video = CompositeVideoClip([clip])

# extract audio
final_audio = final_video.audio

# convert audio to numpy array
sample_rate = 44100
audio_array = final_audio.to_soundarray(fps=sample_rate)

chunk_size = int(sample_rate * 0.1)
rms_values = []

for start_idx in range(0, len(audio_array), chunk_size):
    end_idx = start_idx + chunk_size
    chunk = audio_array[start_idx:end_idx]
    
    if len(chunk) == 0:
        continue
    
    # convert stereo to mono
    chunk_mono = np.mean(chunk, axis=1)

    # RMS = sqrt(mean of squares)
    rms = np.sqrt(np.mean(chunk_mono**2))
    rms_values.append(rms)

# audio spike detection
threshold = np.percentile(rms_values, 95)  # top 5 %
spikes = [i for i, val in enumerate(rms_values) if val > threshold]

spike_times = [i * 0.1 for i in spikes]  # each chunk ~0.1s
rounded_times = np.round(spike_times)

print("Spikes detected at (seconds):")
print(*list(dict.fromkeys(rounded_times)), sep="\n")

# build intervals from spike times
# 5 secs buffer start and end
buffer = 5  
intervals = []

unique_spike_times = list(dict.fromkeys(rounded_times))  # remove duplicates
unique_spike_times.sort()  # sort them in ascending order

for spike_time in unique_spike_times:
    start = max(0, spike_time - buffer)
    end = spike_time + buffer
    intervals.append((start, end))

# merge overlapping intervals
intervals.sort(key=lambda x: x[0])  # sort by start time
merged_intervals = []

if intervals:
    current_start, current_end = intervals[0]
    
    for i in range(1, len(intervals)):
        next_start, next_end = intervals[i]
        
        if next_start <= current_end:
            # Overlaps or touches the current interval
            current_end = max(current_end, next_end)
        else:
            # No overlap: push old interval to the list, start a new one
            merged_intervals.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Add the last interval
    merged_intervals.append((current_start, current_end))

# create subclips for each merged interval
count = 0
for (start, end) in merged_intervals:
    count += 1
    
    highlight = final_video.subclipped(start, end)
    
    filename = f"highlight-{count}.mp4"
    print(f"Writing: {filename} (start={start}, end={end})")
    highlight.write_videofile(filename)
