import numpy as np
import cv2


class VideoUnifier:
    def __init__(self):
        pass

    def _obtain_readers_and_data(self, video_paths):
        vid_readers = list()
        # to measure minimum length of movie
        min_vid_length = 10000000
        # to measure maximum width of movie
        max_width = 0
        # to measure final height
        max_height = 0

        for vid_path in video_paths:
            reader = cv2.VideoCapture("{0}".format(vid_path))
            final_frame_count, frame_width, frame_height, fps = int(
                reader.get(cv2.CAP_PROP_FRAME_COUNT)), int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), reader.get(cv2.CAP_PROP_FPS)
            vid_data_dict = {
                "reader": reader,
                "final_frame_count": final_frame_count,
                "frame_width": frame_width,
                "frame_height": frame_height
            }
            vid_readers.append(vid_data_dict)
            # obtaining the length and width of the result
            if min_vid_length > final_frame_count:
                min_vid_length = final_frame_count
            if max_width < frame_width:
                max_width = frame_width
            if max_height < frame_height:
                max_height = frame_height

        return vid_readers, min_vid_length, max_width, max_height

    def unify_videos(self, videos_to_unify_paths: list, output_file, alignment: str = 'vertical'):

        videos_readers_dicts, min_video_length, max_video_width, max_height = self._obtain_readers_and_data(videos_to_unify_paths)
        if alignment == 'vertical':
            vid_writer = cv2.VideoWriter("{0}".format(output_file), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (max_video_width, max_height*len(videos_readers_dicts)))
        else:
            vid_writer = cv2.VideoWriter("{0}".format(output_file), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                         (max_video_width * len(videos_readers_dicts), max_height))
        frame_ctr = 0

        while frame_ctr < min_video_length:
            current_frames = list()
            for reader_dict in videos_readers_dicts:
                ret, frame = reader_dict['reader'].read()
                if not ret:
                    raise Exception("something went wrong!")
                # resize frame to same width as final movie, and same height as current
                frame = cv2.resize(frame, (max_video_width, max_height))
                current_frames.append(frame)
            if alignment == 'vertical':
                new_composite_frame = np.ndarray(shape=(max_height*len(current_frames), max_video_width, 3))
            else:
                new_composite_frame = np.ndarray(shape=(max_height, max_video_width * len(current_frames), 3))
            # unifying the frames
            if alignment == 'vertical':
                curr_height = 0
                for frame in current_frames:
                    frame_height = frame.shape[0]
                    new_composite_frame[curr_height: curr_height + frame_height, :] = frame
                    curr_height += frame_height
            else:
                curr_width = 0
                for frame in current_frames:
                    frame_width = frame.shape[1]
                    new_composite_frame[:, curr_width: curr_width + frame_width] = frame
                    curr_width += frame_width
            # writing the new frame
            im = np.uint8(new_composite_frame)
            vid_writer.write(im)

            frame_ctr += 1

        # wraping up
        vid_writer.release()
        [reader_dict["reader"].release() for reader_dict in videos_readers_dicts]

        # orig = cv2.LoadImage("rot.png")
        # cv2.Flip(orig, flipMode=-1)

NEW_VID_PATH = "/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/temp/FibrinogenUnifiedNoGradient.avi"
# VIDEO1 = "/Users/yishaiazabary/Downloads/Videos/coll1RBC.avi"
# VIDEO2 = "/Users/yishaiazabary/Downloads/Videos/coll2RBC.avi"
VIDEO1 = "/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/temp/PRP_FIBRINOGEN 2.avi"
VIDEO2 = "/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/temp/PRP_FBG_exp.63_control_SP3.avi"
VIDEO3 = "/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/temp/PRP_FBG_exp.63_control_SP4.avi"
vu = VideoUnifier()
vu.unify_videos([VIDEO1, VIDEO2, VIDEO3], NEW_VID_PATH, alignment='horizontal')
