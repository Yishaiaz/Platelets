import os
import numpy as np
import time as time
import cv2

TEST_FILE_NAME = "exp89_PRP_FBG_CONTROL01_R3D_SP1.avi"


# attach_color_scheme = np.array([[0, 0, 240],   [1, 1, 230],
#                                 [2, 2, 220],   [3, 3, 210],
#                                 [4, 4, 200],   [5, 5, 190],
#                                 [6, 6, 180],   [7, 7, 170],
#                                 [8, 8, 160],   [9, 9, 150],
#                                 [10, 10, 140], [11, 11, 130],
#                                 [12, 12, 120], [13, 13, 110],
#                                 [14, 14, 100], [15, 15, 90],
#                                 [16, 16, 80],  [17, 17, 70],
#                                 ])
# detach_color_scheme = np.array([[240, 0, 0],   [230, 1, 1],
#                                 [220, 2, 2],  [210, 3, 3],
#                                 [200, 4, 4],  [190, 5, 5],
#                                 [180, 6, 6],  [170, 7, 7],
#                                 [160, 8, 8],  [150, 9, 9],
#                                 [140, 10, 10], [130, 11, 11],
#                                 [120, 12, 12], [110, 13, 13],
#                                 [100, 14, 14], [90,  15, 15],
#                                 [80,  16, 16], [70,  17, 17],
#                                 ])

attach_color_scheme = np.array([[0, 0, 250],   [0, 0, 250],
                                [0, 0, 250],   [0, 0, 250],
                                [0, 0, 250],   [0, 0, 250],
                                [0, 0, 250],   [0, 0, 250],
                                [0, 0, 250],   [0, 0, 250],
                                [0, 0, 250], [0, 0, 250],
                                [0, 0, 250], [0, 0, 250],
                                [0, 0, 250], [0, 0, 250],
                                [0, 0, 250],  [0, 0, 250],
                                ])
detach_color_scheme = np.array([[250, 0, 0],   [250, 0, 0],
                                [250, 0, 0],  [250, 0, 0],
                                [250, 0, 0],  [250, 0, 0],
                                [250, 0, 0],  [250, 0, 0],
                                [250, 0, 0],  [250, 0, 0],
                                [250, 0, 0], [250, 0, 0],
                                [250, 0, 0], [250, 0, 0],
                                [250, 0, 0], [250,  0, 0],
                                [250,  0, 0], [250,  0, 0],
                                ])



# attach_color_scheme = np.array([[0, 0, 240],   [10, 1, 230],
#                                 [20, 2, 220],   [30, 3, 210],
#                                 [40, 4, 200],   [50, 5, 190],
#                                 [60, 6, 180],   [70, 7, 170],
#                                 [80, 8, 160],   [90, 9, 150],
#                                 [100, 10, 140], [110, 11, 130],
#                                 [120, 12, 120], [130, 13, 110],
#                                 [140, 14, 100], [150, 15, 90],
#                                 [160, 16, 80],  [170, 17, 70],
#                                 ])
# detach_color_scheme = np.array([[240, 0,  0],   [230, 1,  10],
#                                 [220, 2,  20],  [210, 3,  30],
#                                 [200, 4,  40],  [190, 5,  50],
#                                 [180, 6,  60],  [170, 7,  70],
#                                 [160, 8,  80],  [150, 9,  90],
#                                 [140, 10, 100], [130, 11, 110],
#                                 [120, 12, 120], [110, 13, 130],
#                                 [100, 14, 140], [90,  15, 150],
#                                 [80,  16, 160], [70,  17, 170],
#                                 ])

class videoDynamicsEncoder:

    def __init__(self, color_scheme1: np.array([]), color_scheme2: np.array = np.array([]), **kwargs):
        """

        :param color_scheme1: attach events
        :param color_scheme2: detach events
        :param kwargs:
        which_to_color: {'both', 'attach', 'detach'}
        """
        if kwargs.get('which_to_color') is not None:
            self.which_to_color = kwargs.get('which_to_color')
        else:
            self.which_to_color = 'both'

        self.attach_color_scheme = color_scheme1
        self.detach_color_scheme = color_scheme2

    def color_by_scheme(self, buf: np.ndarray, dynamics: np.ndarray, limit, top_limit=30):
        # detach
        if limit < 0:
            for i in range(len(self.detach_color_scheme)-1):
                color = self.detach_color_scheme[i]
                buf = np.where((dynamics < limit - i) & (dynamics >= limit - (i+1)), color, buf)
                last_ctr = i
            buf = np.where(dynamics < limit - last_ctr-1, self.detach_color_scheme[len(self.detach_color_scheme)-1], buf)
        # attach
        elif limit > 0:
            for i in range(len(self.attach_color_scheme)-1):
                color = self.attach_color_scheme[i]
                buf = np.where((dynamics > limit + i) & (dynamics <= limit + (i+1)), color, buf)
                last_ctr = i
            buf = np.where(dynamics > limit + last_ctr+1, self.attach_color_scheme[len(self.attach_color_scheme)-1], buf)

        return buf

    def manipulate_frame(self, buf:np.ndarray, **kwargs):
        """
        colors are [Blue,Green,Red]
        :param buf:
        :return:
        """
        if kwargs.get('limit') is not None:
            limit = kwargs.get('limit')
        else:
            limit = -9

        dynamics = buf[0]-buf[1]
        if self.which_to_color == 'both':
            # attach
            buf[0] = self.color_by_scheme(buf[0], dynamics, (-1)*limit, top_limit=30)
            # detach
            buf[0] = self.color_by_scheme(buf[0], dynamics, limit, top_limit=-30)
        elif self.which_to_color == 'detach':
            buf[0] = self.color_by_scheme(buf[0], dynamics, limit, top_limit=-30)
        else:
            buf[0] = self.color_by_scheme(buf[0], dynamics, (-1)*limit, top_limit=30)
        return buf

    def manipulate_video(self, video_path: str, manipulated_video_path, condition, **kwargs):
        """
        condition is a function type object that receives pixel
        dynamic value (current_value-next_frame_value),current pixel time, current intensity.
        :param condition:
        :return:
        """
        video_cap = cv2.VideoCapture("{0}".format(video_path))
        # get video meta data
        final_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frames_in_memory = 2
        # video writer to AVI
        out = cv2.VideoWriter(
            "{0}".format(manipulated_video_path),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        # vid_writer = cv2.VideoWriter("videos/suspicious_dynamics_videoes/{0}".format(video_name), cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V'), fps, (frame_width, frame_height))

        buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
        ret = True

        fc = 0
        while fc < final_frame_count and ret:
            # loading the data according
            in_memory_frames_ctr = 0
            single_frame_start_time = time.time()
            while in_memory_frames_ctr < frames_in_memory:
                if np.sum(buf[frames_in_memory - 1]) > 0:
                    temp = buf[frames_in_memory - 1].copy()
                    # buf[FRAMES_IN_MEMORY-1] = np.zeros((frame_height, frame_width, 3))
                    buf = np.zeros((frames_in_memory, frame_height, frame_width, 3), np.dtype('int64'))
                    buf[0] = temp
                else:
                    ret, frame = video_cap.read()
                    buf[in_memory_frames_ctr] = frame
                    fc += 1
                in_memory_frames_ctr += 1
            buf = condition(buf, **kwargs)
            im = np.uint8(buf[0])
            out.write(im)
            # cv2.imshow("test", np.array(im, dtype=np.uint8))
            # new_video[fc-2] = buf[0]
            single_frame_end_time = time.time()
            print("frame No:{0} has been manipulated.took:{1:.2} seconds".format(fc - 2,
                                                                                 single_frame_end_time - single_frame_start_time))

        out.release()
        video_cap.release()
        total_end = time.time()
        # print("{0}: {1:.2f} minutes total".format(fc, (total_end - total_start) / 60))


if __name__ == "__main__":
    # color_scheme1 = np.array(['#FFA500'])
    # color_scheme2 = np.array(['#0000FF'])
    main_directory = os.fsencode("ForAnalyze/Final/")
    videoDynamicsEncoder = videoDynamicsEncoder(attach_color_scheme, detach_color_scheme)
    for file in os.listdir(main_directory):
        file_name = os.fsdecode(file)
        if file_name.__contains__("avi"):
            if file_name.__contains__("asdf"):
                videoDynamicsEncoder.manipulate_video("ForAnalyze/Final/" + file_name,
                                                      "videos/suspicious_dynamics_videoes/" + file_name,
                                                      videoDynamicsEncoder.manipulate_frame, limit=-14)
            else:
                videoDynamicsEncoder.manipulate_video("ForAnalyze/Final/"+file_name, "ForAnalyze/temp/"+file_name, videoDynamicsEncoder.manipulate_frame, limit=-12)

