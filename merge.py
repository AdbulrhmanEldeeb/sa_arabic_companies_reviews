from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load the video clips
video1 = VideoFileClip("code_run_1.wmv")
video2 = VideoFileClip("gradio.wmv")

# Concatenate the videos
final_clip = concatenate_videoclips([video1, video2])

# Write the result to a file
final_clip.write_videofile("merged.mp4")