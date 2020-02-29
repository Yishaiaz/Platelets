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

    def unify_videos(self, videos_to_unify_paths: list, output_file):

        videos_readers_dicts, min_video_length, max_video_width, max_height = self._obtain_readers_and_data(videos_to_unify_paths)

        vid_writer = cv2.VideoWriter("{0}".format(output_file), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (max_video_width, max_height*len(videos_readers_dicts)))

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

            new_composite_frame = np.ndarray(shape=(max_height*len(current_frames), max_video_width, 3))
            # unifying the frames
            curr_height = 0
            for frame in current_frames:
                frame_height = frame.shape[0]
                new_composite_frame[curr_height: curr_height + frame_height, :] = frame
                curr_height += frame_height

            # writing the new frame
            im = np.uint8(new_composite_frame)
            vid_writer.write(im)

            frame_ctr += 1

        # wraping up
        vid_writer.release()
        [reader_dict["reader"].release() for reader_dict in videos_readers_dicts]

        # # vid_reader1 = cv2.VideoCapture("{0}".format(VIDEO1))
        # # vid_reader2 = cv2.VideoCapture("{0}".format(VIDEO2))
        # # vid1_final_frame_count, vid1_frame_width, vid1_frame_height, vid1_fps = int(
        # #     vid_reader1.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_reader1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        # #     vid_reader1.get(cv2.CAP_PROP_FRAME_HEIGHT)), vid_reader1.get(cv2.CAP_PROP_FPS)
        # # vid2_final_frame_count, vid2_frame_width, vid2_frame_height, vid2_fps = int(
        # #     vid_reader2.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_reader2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        # #     vid_reader2.get(cv2.CAP_PROP_FRAME_HEIGHT)), vid_reader2.get(cv2.CAP_PROP_FPS)
        # new_vid_num_frames = min(vid1_final_frame_count, vid2_final_frame_count)
        # new_vid_width = max(vid1_frame_width, vid2_frame_width)
        # new_vid_height = vid1_frame_height + vid2_frame_height
        # new_vid_shape = (new_vid_num_frames, new_vid_width, new_vid_height, 3)
        # new_vid = np.ndarray(shape=new_vid_shape)
        # out = cv2.VideoWriter(
        #     "{0}".format(NEW_VID_PATH),
        #     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (new_vid_width, new_vid_height))
        # fc = 0
        # ret = True
        # while fc < new_vid_num_frames and ret:
        #
        #     ret1, frame_upper_part = vid_reader1.read()
        #     ret2, frame_lower_part = vid_reader2.read()
        #     ret = ret1 * ret2
        #     if ret:
        #         frame_upper_part = cv2.resize(frame_upper_part, (new_vid_width, vid1_frame_height), fx=0, fy=0,
        #                                       interpolation=cv2.INTER_CUBIC)
        #         frame_lower_part = cv2.resize(frame_lower_part, (new_vid_width, vid2_frame_height), fx=0, fy=0,
        #                                       interpolation=cv2.INTER_CUBIC)
        #         new_composite_frame = np.ndarray(shape=(new_vid_height, new_vid_width, 3))
        #         new_composite_frame[:vid1_frame_height, :] = frame_upper_part
        #         new_composite_frame[vid1_frame_height:, :] = frame_lower_part
        #         im = np.uint8(new_composite_frame)
        #         out.write(im)
        #     fc += 1
        #     print(fc)
        #
        # out.release()
        # vid_reader1.release()
        # vid_reader2.release()

NEW_VID_PATH = "/Users/yishaiazabary/Downloads/Videos/new_vid.avi"
VIDEO1 = "/Users/yishaiazabary/Downloads/Videos/media4.avi"
VIDEO2 = "/Users/yishaiazabary/Downloads/Videos/media3.avi"
vu = VideoUnifier()
vu.unify_videos([VIDEO1, VIDEO2], NEW_VID_PATH)