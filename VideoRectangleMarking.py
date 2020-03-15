import numpy as np
import cv2 as cv


def add_rec_in_frame(frame: np.array, rec_top_left_corner: tuple = (0,0), rec_bottom_right_corner: tuple=(1,1), color: tuple = (255, 0, 0), perim_thickness: int = 1):
    return cv.rectangle(frame, rec_top_left_corner, rec_bottom_right_corner, color, perim_thickness)


def process_video(video_path, rec_per_frame_list: [list, tuple] = ((0, 0), (1, 1))):
    """

    :param video_path:
    :param rec_per_frame_list: list of [or a single] tuple of tuples, ( (starting point x, starting point y), (end point x, end point y))
    :return:
    """
    video_cap = cv.VideoCapture(video_path)
    final_frame_count = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv.VideoWriter(
                "{0}".format(video_path[:-4]+"with_rectangle.avi"),
                cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    ret = True
    fc = 0
    while ret and fc < final_frame_count:
        ret, frame = video_cap.read()
        rec_tuples = None
        if type(rec_per_frame_list) is list:
            rec_tuples = rec_per_frame_list[fc]
        else:
            rec_tuples = rec_per_frame_list
        frame = add_rec_in_frame(frame=frame, rec_top_left_corner=rec_tuples[0], rec_bottom_right_corner=rec_tuples[1])
        video_writer.write(frame)
        fc += 1


process_video("ForAnalyze/temp/PLT_coll4_exp.63_control_SP2.avi", ((0, 0), (20, 20)))
