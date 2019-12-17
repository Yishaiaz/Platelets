import numpy as np
import cv2 as cv
from librosa import feature as libzcr
from ToTimeSeries import ToTimeSeries as tts
from InputReader import Simple_Input_Reader as sir


class ZeroCrossingRateFeature:
    def calc_dynamics_vector(self, vector):
        dynamics_vector = np.zeros((len(vector)-1, ))
        for i in range(0, len(dynamics_vector)-1):
            dynamics_vector[i] = np.average(vector[i]) - np.average(vector[i+1])
        return dynamics_vector

    def split_into_bins(self, stack_of_images: np.ndarray, bin_size: int = 4):
        bin_transformer = tts(bin_size=bin_size, original_file=stack_of_images, frame_count=len(stack_of_images), single_frame_width=len(stack_of_images[1]), single_frame_height=len(stack_of_images[1][1]))
        return bin_transformer.into_time_series()

    def calc_zcr_per_bin(self, single_bin_stack: np.ndarray, time_intervals: int = 5):
        zcr_per_bin = np.zeros((int(len(single_bin_stack)/time_intervals), ))
        for i in range(0, len(zcr_per_bin)):
            j = i * time_intervals
            bins_stack_per_interval = single_bin_stack[j:j+time_intervals]
            zcr_per_bin[int(j/time_intervals)] = libzcr.zero_crossing_rate(self.calc_dynamics_vector(bins_stack_per_interval))
        return zcr_per_bin

    def calc_zcr_for_all_bins(self, all_bin_stack, time_intervals: int = 5):
        zcr_list_per_bin = list()
        for bin_ctr in range(0, len(all_bin_stack)):
            zcr_list_per_bin.append(self.calc_zcr_per_bin(all_bin_stack[bin_ctr], time_intervals=time_intervals))

        return zcr_list_per_bin


sir = sir()
zcr = ZeroCrossingRateFeature()
vid_stack, frame_number, frame_width, frame_height = sir.input_to_np("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/PlateletInteractionCloserSide.avi")
vid_stack_in_bins = zcr.split_into_bins(vid_stack)
# print(zcr.calc_zcr_per_bin(vid_stack_in_bins[0], 5))
print(zcr.calc_zcr_for_all_bins(vid_stack_in_bins))


