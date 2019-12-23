import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ToTimeSeries import ToTimeSeries as tts
from InputReader import Simple_Input_Reader as sir


class DynamicsDistributionOverTime:

    def __init__(self,  **kwargs):
        if kwargs.get('time_frame_size'):
            time_frame_size = kwargs.get('time_frame_size')
        else:
            time_frame_size = 10
        self.TIME_FRAME_SIZE = time_frame_size
        self.DYNAMICS_RANGE = np.linspace(-30, 30, 61)

    def calc_dynamics_distribution_for_single_time_frame(self, time_frame_data, **kwargs):
        """
        KWARGS:
        normalize_by_max : boolean, default is 'True'
        :param time_frame_data:
        :param kwargs:
        :return:
        """
        # COUNT DELTAS
        deltas_counters = np.zeros((len(self.DYNAMICS_RANGE),))
        for frame_ctr in range(0, len(time_frame_data) - 1):
            deltas_between_current_frames = (time_frame_data[frame_ctr].flatten() - time_frame_data[frame_ctr + 1].flatten())#.flatten()
            # count appearance of each delta
            for delta_ctr in range(0, len(deltas_counters)):
                deltas_counters[delta_ctr] += len(deltas_between_current_frames[deltas_between_current_frames == self.DYNAMICS_RANGE[delta_ctr]])

        # NORMALIZE DELTA COUNTERS
        if kwargs.get('normalize_by_max') is not None:
            normalize_by_max = kwargs.get('normalize_by_max')
        else:
            normalize_by_max = True
        if not normalize_by_max:
            deltas_counters = deltas_counters / np.sum(deltas_counters)
        else:
            deltas_counters = deltas_counters / np.max(deltas_counters)

        return deltas_counters

    def calc_dynamics_distribution(self, file_path, **kwargs):
        sir_instance = sir()
        vid_stack, frame_number, frame_width, frame_height = sir_instance.input_to_np(file_path)

        self.number_of_time_frames = int(frame_number/self.TIME_FRAME_SIZE)
        time_frame_dynamics_distributions = np.zeros((self.number_of_time_frames, len(self.DYNAMICS_RANGE)))

        time_frame_array = list()
        time_frame_start_range = 0
        time_frame_end_range = self.TIME_FRAME_SIZE

        for time_frame_ctr in range(0, self.number_of_time_frames):
            if time_frame_start_range > len(vid_stack): break
            if time_frame_end_range > len(vid_stack):
                time_frame_end_range = len(vid_stack)
            time_frame_array.append(vid_stack[time_frame_start_range: time_frame_end_range, :, :, 0].copy().astype('float'))
            time_frame_end_range+=self.TIME_FRAME_SIZE
            time_frame_start_range += self.TIME_FRAME_SIZE

        for time_frame_ctr in range(0, len(time_frame_array)):
            time_frame_dynamics_distributions[time_frame_ctr] = self.calc_dynamics_distribution_for_single_time_frame(time_frame_array[time_frame_ctr], **kwargs)

        return time_frame_dynamics_distributions



ddot = DynamicsDistributionOverTime()
to_plot = ddot.calc_dynamics_distribution("/Users/yishaiazabary/PycharmProjects/platelets/ForAnalyze/PLT_coll4_exp.63_control_SP1.avi", normalize_by_max=True)
to_plot[:, 31:32] = 0
# to_plot[0:2] = 0
col_labels = []
for i in range(0, len(ddot.DYNAMICS_RANGE)):
    if i % 2 == 0:
        col_labels.append(ddot.DYNAMICS_RANGE[i])
row_labels = ["{0}".format(x * ddot.TIME_FRAME_SIZE) for x in range(0, ddot.number_of_time_frames)]
sns.set()
ax = sns.heatmap(data=to_plot, vmax=0.1)
ax.set_yticklabels(labels=row_labels, rotation=0)
ax.set_xticklabels(labels=col_labels, rotation=45)
plt.show()




