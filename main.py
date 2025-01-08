from moviepy import VideoFileClip, TextClip, CompositeVideoClip
import numpy as np


# uses subclipped to select seconds within the video.
clip = (
    VideoFileClip("liverpool-vs-united.mp4")
    )


# NOTE: allows text to be added over the video clip

# txt_clip = TextClip(
#     font="Arial.ttf",
#     text="",
#     font_size=80,
#     color='white'
# ).with_duration(10).with_position('center')


# creates video
final_video = CompositeVideoClip([clip])
# final_video.write_videofile("result.mp4")

# converts to audio
final_audio = final_video.audio

# Peak detection

sample_rate = 44100
audio_array = final_audio.to_soundarray(fps=sample_rate)

chunk_size = int(sample_rate * 0.1)
rms_values = []

for start in range(0, len(audio_array), chunk_size):
    end = start + chunk_size
    chunk = audio_array[start:end]
    
    # If there's not enough samples (last chunk?), skip
    if len(chunk) == 0:
        continue
    
    # Convert stereo to mono
    chunk_mono = np.mean(chunk, axis=1)  # shape: (chunk_length,)

    # RMS = sqrt(mean of squares)
    rms = np.sqrt(np.mean(chunk_mono**2))
    rms_values.append(rms)

# threshold
threshold = np.percentile(rms_values, 95) 
spikes = [i for i, val in enumerate(rms_values) if val > threshold]

spike_times = [i * 0.1 for i in spikes]  # each chunk is ~0.1s


rounded_times = np.round(spike_times)
print("Spikes detected at (seconds):")
print(*list(dict.fromkeys(rounded_times)), sep="\n")


buffer = 2  # how many seconds before & after the spike
count = 0


# FIXME: still new work
# Remove duplicates
unique_spike_times = list(dict.fromkeys(rounded_times))

for spike_time in unique_spike_times:
    start = max(0, spike_time - buffer)
    end = spike_time + buffer
    
    highlight = final_video.subclipped(start, end)
    
    # name the file something like highlight-1.mp4
    filename = f"highlight-{count}.mp4"
    highlight.write_videofile(filename)
    
    count += 1
