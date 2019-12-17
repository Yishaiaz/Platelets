from ToTimeSeries import ToTimeSeries
from SimpleVisualizationTool import simpleVisualization as sv

tts = ToTimeSeries(bin_size=90, y_size=90)
result = tts.into_time_series()
bin_num = 3
frame_num = 60
print(result[bin_num][frame_num][0][0])
video = result[bin_num]
sv = sv()
sv.visualize_video(video=video, frame_update_frequency=0.5, gray=False, frame_title='test title')
