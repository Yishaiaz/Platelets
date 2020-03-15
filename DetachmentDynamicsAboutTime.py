import os
from math import factorial
# import pandas as pd
import numpy as np
import cv2
# from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns


class DetachmentDynamicsAboutTime:

    def __init__(self, **kwargs):
        """
        threshold for events is the positive integer threshold, the negative one will be auto calculated.
        :param kwargs:
        """
        if kwargs.get('threshold_for_events') is not None:
            self.threshold_for_events = kwargs.get('threshold_for_events')
        else:
            self.threshold_for_events = 10
        
        if kwargs.get('smoothing_enabled') is not None:
            self.smoothing_enabled = kwargs.get('smoothing_enabled')
        else:
            self.smoothing_enabled = (True, 12)

        if kwargs.get('calc_attachments') is not None:
            self.calc_attachments = kwargs.get('calc_attachments')
        else:
            self.calc_attachments = False
        self.DYNAMICS_RANGE = np.linspace(-30, 30, 61)


    def get_number_of_event(self, dynamics_matrix: np.ndarray):
        detachment_event = dynamics_matrix[dynamics_matrix >= self.threshold_for_events]
        attachment_event = dynamics_matrix[dynamics_matrix < -1 * self.threshold_for_events]
        if self.calc_attachments:
            return len(attachment_event) / (len(detachment_event) + len(attachment_event)) if len(
                attachment_event) > 0 else 0
        else:
            return len(detachment_event) / (len(detachment_event) + len(attachment_event)) if len(attachment_event)>0 else 0

    def count_detachment_events(self, video_path: str):
        # reading the file
        video_cap = cv2.VideoCapture(video_path)
        final_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        distribution_for_detachment_event_about_time = np.ndarray((final_frame_count - 1), dtype=float)

        ret = True
        fc = 0
        first_frame = None
        while fc < final_frame_count - 1 and ret:
            if first_frame is None:
                ret, first_frame = video_cap.read()
                ret, second_frame = video_cap.read()
            else:
                ret, second_frame = video_cap.read()
            dynamics_matrix = second_frame.astype(float)[:, :, 0] - first_frame.astype(float)[:, :, 0]
            distribution_for_detachment_event_about_time[fc] = self.get_number_of_event(dynamics_matrix)
            first_frame = second_frame.copy()
            fc += 1

        return distribution_for_detachment_event_about_time if not self.smoothing_enabled[0] else \
            self.savitzky_golay_smoothing(distribution_for_detachment_event_about_time, window_size=self.smoothing_enabled[1], order=3)

    def savitzky_golay_smoothing(self, y, window_size=5, order=3, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError as msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    def single_signal_plot(self, to_plot: np.ndarray):
        fig, ax = plt.subplots()
        # to_plot = only_prp_arrays[idx]
        time_intervals = np.linspace(0, len(to_plot) / 7, len(to_plot))
        ax.set(title='A single platelet on fibrinogen')
        ax.plot(time_intervals, to_plot)
        # ax.plot([0,14], [0.5,0.5], )
        plt.yticks(np.arange(0, 1.1, step=0.1), [str(int(x * 10) / 10) for x in np.arange(0, 1.1, step=0.1)])
        ax.set_xlabel("time (sec)", fontsize=5)
        ax.set_ylabel("#Detachment Events/ # All Evenets", fontsize=5)
        ax.grid()
        plt.show()
        # fig.savefig("for Bennys.png", dpi=300)

    def combined_plots_for_substrates(self, first_type_arrays: np.ndarray, second_type_arrays: np.ndarray):
        """

        :param first_type_arrays: a 2D array, each row is a different signal of a platelet spreading process.
        :param second_type_arrays: a 2D array, each row is a different signal of a platelet spreading process.
        :return:
        """
        ylim = (0, 1)
        fig, axis = plt.subplots(2, 1)
        for i, val in enumerate(first_type_arrays):
            time_intervals = np.linspace(0,len(val)/7, len(val))
            axis[0].plot(time_intervals, val)
            axis[0].set_xlabel("time (sec)", fontsize=8)
            axis[0].set_ylabel("#Detachment Events/ # All Evenets", fontsize=6)
        for i, val in  enumerate(second_type_arrays):
            time_intervals = np.linspace(0,len(val)/7, len(val))
            axis[1].plot(time_intervals, val)
            axis[1].set_xlabel("time (sec)", fontsize=8)
            axis[1].set_ylabel("#Detachment Events/ # All Evenets", fontsize=6)
        plt.setp(axis, ylim=ylim)
        plt.tight_layout()
        plt.show()
        # fig.savefig("DetachmentEventPlots/ConcentratedPlotsBySubstrate.png", dpi=300, bbox_inches="tight")

    def combined_plots_for_substrates_outlier_events(self, to_plot: np.ndarray):
        """

        :param to_plot: a 2D array, each row is a different signal of a platelet spreading process.
        :return:
        """
        ylim = (0, 15)
        fig, ax = plt.subplots()
        for i, val in enumerate(to_plot):
            # take the last 15% of a signal
            percent_of_last_frames = 0.85
            last_frames = val[int(len(val) * percent_of_last_frames):]
            rest_of_frames = val[:int(len(val) * percent_of_last_frames)]
            last_frames_mean = np.mean(last_frames)
            last_frames_std = np.std(last_frames)
            # for each 10 frames, count the number of events beyond mean+-std
            results = np.zeros(shape=(int(len(rest_of_frames) / 10) - 1), dtype=int)
            for frame_iterator in range(0, int(len(rest_of_frames) / 10) - 1):
                frames_to_count = rest_of_frames[frame_iterator * 10:(frame_iterator + 1) * 10]
                mask = (frames_to_count > (last_frames_mean + last_frames_std)) | (
                            frames_to_count < (last_frames_mean - last_frames_std))
                number_of_outlier_events = len(frames_to_count[mask])
                results[frame_iterator] = number_of_outlier_events
            # plot the results
            t = np.linspace(0, len(results), len(results))
            ax.plot(t, results, 'o', label='{} {}'.format(i, 'FBR'))
            ax.set_xlabel('time (frame No. /10)')
            ax.set_ylabel('number of outlier events')
        # show plot
        ax.set(title='number of outlier events')
        ax.grid()
        plt.setp(ax, ylim=ylim)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # fig.savefig('for_benny.png', dpi=300)


if __name__=="__main__":
    sns.set(style="ticks")
    nrows = 3
    ncols = 2
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, sharey='all')
    directory_name = "ForAnalyze/Final"
    directory = os.fsencode(directory_name)
    dda = DetachmentDynamicsAboutTime(threshold_for_events=12, smoothing_enabled=(True, 17), calc_attachments=True)
    detachment_distributions_array = list()
    detachment_distributions_array_FBR = list()
    detachment_distributions_array_COLL = list()
    file_names_list = list()
    video_length_list_in_seconds = list()
    row_ctr = 0
    col_ctr = 0
    ctr = 0
    for i, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename == ".DS_Store" or filename == "temp":
            continue
        video_detachment_distribution_about_time = None
        try:
            video_detachment_distribution_about_time = dda.count_detachment_events(directory_name + os.sep + filename)
        except Exception as e:
            print(e)
            continue

        detachment_distributions_array.append(video_detachment_distribution_about_time)
        file_names_list.append(filename[:filename.find(' .avi')-2])
        video_length_list_in_seconds.append(len(video_detachment_distribution_about_time)/7) # fps = 7 in all samples
    # fig, ax = plt.subplots()
    # barplot1 = ax.boxplot(box_plots_data, vert=True, patch_artist=True, labels=file_names_list)
    # ax.tick_params(axis='both', which='both', labelsize=4)#, labelrotation=45.0)
    # # ax.tick_params(axis='both', which='both', labelsize=2, labelrotation=0.1)
    # ax.set_title('comparison')
    # ax.yaxis.grid(True)
    # plt.show()
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    # # plt.gcf().subplots_adjust(bottom=0.50)
    # # rcParams.update({'figure.autolayout': True})
    # fig.savefig("DetachmentEventPlots/AllVideosBoxPlot.png", dpi=300, bbox_inches="tight")
    # plt.tight_layout()
    # plt.show()
        if filename.__contains__("PRP"):
            detachment_distributions_array_FBR.append(video_detachment_distribution_about_time)
        elif filename.__contains__("COLLAGEN4"):
            detachment_distributions_array_COLL.append(video_detachment_distribution_about_time)

    # row_ctr = 0
    # col_ctr = 0
    # ylim = (0, 1)
    # detachment_distributions_array = np.array(detachment_distributions_array)
    # for i, val in enumerate(detachment_distributions_array):
    #     if row_ctr - nrows >= 0:
    #         row_ctr = 0
    #         col_ctr += 1
    #
    #     axis[row_ctr][col_ctr].plot(np.linspace(0, video_length_list_in_seconds[i], len(val)), val)
    #     axis[row_ctr][col_ctr].set(title='{}'.format(file_names_list[i].replace('_', ' ', -1)))
    #     axis[row_ctr][col_ctr].grid()
    #     axis[row_ctr][col_ctr].tick_params(axis='y', which='both', labelsize=4, labelrotation=0.1)
    #     axis[row_ctr][col_ctr].title.set_fontsize(5)
    #     if col_ctr == 0:
    #         axis[row_ctr][col_ctr].set_ylabel("#Detachment Events/ # All Evenets", fontsize=4)
    #     if row_ctr == nrows - 1:
    #         axis[row_ctr][col_ctr].set_xlabel("time (sec)", fontsize=4)
    #     row_ctr += 1
    # plt.setp(axis, ylim=ylim)
    # plt.tight_layout()
    # plt.show()
    # fig.savefig("DetachmentEventPlots/AllVideosBoxedPlots.png", dpi=300, bbox_inches="tight")
    np.save("allAttachmentArrays", np.array(detachment_distributions_array))
    # np.save("PRPAttachmentArrays", np.array(detachment_distributions_array_FBR))
    # np.save("PRPNoiseAttachmentArrays", np.array(detachment_distributions_array_FBR))
    # np.save("COLLAttachmentArrays", np.array(detachment_distributions_array_COLL))
    # np.save("allDetachmentArrays", np.array(detachment_distributions_array))
    # np.save("PRPDetachmentArrays", np.array(detachment_distributions_array_FBR))
    # np.save("PRPNoiseDetachmentArrays", np.array(detachment_distributions_array_FBR))
    # np.save("COLLDetachmentArrays", np.array(detachment_distributions_array_COLL))
    # np.save("COLLBACKGROUNDArrays", np.array(detachment_distributions_array))
    # np.save("FibrinogenActingLikeCollagenAttachment", np.array(detachment_distributions_array))
    # np.save("FibrinogenActingLikeCollagenDetachment", np.array(detachment_distributions_array))
    # np.save("tempPRPDetachmentArrays", np.array(detachment_distributions_array))
    # np.save("tempCOLLDetachmentArrays", np.array(detachment_distributions_array))
    # DetachmentDynamicsAboutTime().single_signal_plot()