from moviepy import VideoFileClip, TextClip, CompositeVideoClip


clip = (
    VideoFileClip("liverpool-vs-united.mp4")
    .subclipped(5, 18)
    )


txt_clip = TextClip(
    font="Arial.ttf",
    text="PLAY VISION",
    font_size=80,
    color='white'
).with_duration(10).with_position('center')


final_video = CompositeVideoClip([clip, txt_clip])
final_video.write_videofile("result.mp4")
