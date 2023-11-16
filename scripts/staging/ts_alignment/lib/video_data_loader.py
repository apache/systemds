import numpy as np
import cv2
import datetime as dt
import pandas as pd

from abc import ABC, abstractmethod
from sklearn import preprocessing
from data_loader import DataLoader
from video_sampling import VideoSampling


class ExtractionMethod(ABC):
    def __init__(self, column_names, column_type):
        self.data_loader = None
        self.column_names = column_names
        self.column_type = column_type

    @abstractmethod
    def extract(self, frame, timestamp):
        raise 'Not implemented for this extraction method'

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def setup_df(self):
        df = pd.DataFrame(columns=[self.column_names], index=pd.to_datetime([]))
        df[self.column_names] = df[self.column_names].astype(self.column_type)
        df.index.name = 'timestamp'
        return df


class PixelExtractor(ExtractionMethod):
    def __init__(self):
        super().__init__('pixel_values', column_type=object)
        self.frame_needs_resizing = False

    def set_data_loader(self, data_loader):
        super().set_data_loader(data_loader)

        if data_loader.x is not None:
            if data_loader.frame_width != data_loader.x.frame_width or data_loader.frame_height != data_loader.x.frame_height:
                self.frame_needs_resizing = True

    def extract(self, frame, timestamp):
        if self.frame_needs_resizing and (
                self.data_loader.sampling_strategy is None or not self.data_loader.sampling_strategy.downscale):
            frame = cv2.resize(frame, (self.data_loader.x.frame_width, self.data_loader.x.frame_height),
                               fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        pixels = np.dstack((frame, np.zeros(frame.shape[:2], 'uint8'))).view('uint32').squeeze(-1).flatten()
        self.data_loader.df.loc[timestamp, self.column_names] = np.float32(
            preprocessing.normalize([pixels]).reshape(-1))


class IntensityExtractor(ExtractionMethod):
    def __init__(self):
        super().__init__('intensity', column_type=float)

    def extract(self, frame, timestamp):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.
        self.data_loader.df.loc[timestamp, self.column_names] = frame.mean()


class SiftFeatureExtractor(ExtractionMethod):
    def __init__(self, num_features):
        super().__init__('sift_features', column_type=object)

        self.num_features = num_features

    def extract(self, frame, timestamp):
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        kp, des = sift.detectAndCompute(src_gray, None)

        result = np.full((self.num_features, 128), -1)

        if des is not None:
            shape = des.shape[0]
            result[0:des.shape[0] if shape < self.num_features else self.num_features] = des \
                if shape < self.num_features else des[0:self.num_features]

        self.data_loader.df.loc[timestamp, self.column_names] = preprocessing.normalize([result.flatten()]).reshape(-1)


class OrbFeatureExtractor(ExtractionMethod):
    def __init__(self, num_features=2000):
        super().__init__('orb_features', column_type=object)
        self.num_features = num_features

    def extract(self, frame, timestamp):
        orb = cv2.ORB_create(nfeatures=self.num_features)
        original_keypoints, original_descriptor = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

        result = np.full((self.num_features, 32), -1)
        if original_descriptor is not None:
            shape = original_descriptor.shape[0]
            result[0:original_descriptor.shape[
                0] if shape < self.num_features else self.num_features] = original_descriptor \
                if shape < self.num_features else original_descriptor[0:self.num_features]

        self.data_loader.df.loc[timestamp, self.column_names] = preprocessing.normalize([result.flatten()]).reshape(-1)


class HistogramExtractor(ExtractionMethod):
    def __init__(self, num_bins):
        super().__init__([str(i) for i in range(0, num_bins)], column_type=float)
        self.num_bins = num_bins

    def extract(self, frame, timestamp):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        if self.data_loader.sampling_strategy is not None:
            mask = self.data_loader.sampling_strategy.mask

        histogram = cv2.calcHist([hsv], [0], mask, [self.num_bins], [0, 180])
        histogram = cv2.normalize(histogram, frame, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

        self.data_loader.df.loc[timestamp, self.column_names] = histogram


class VideoLoader(DataLoader):
    def __init__(self, path: str, batch_size: int, extraction_method: ExtractionMethod, start_time, x=None):
        super().__init__(batch_size, x)
        self.sampling_strategy = None
        self.overall_frame_counter = 0

        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise 'Video can not be read!'

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.frame_start_position = 0
        self.start_time = start_time
        self.extraction_method = extraction_method
        self.extraction_method.set_data_loader(self)
        self.set_properties()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def set_properties(self):
        self.set_aligned_start_time()
        self.all_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.set_batch_size(int(self.batch_size) if self.batch_size is not None else self.all_frames)
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.end_time = self.start_time + pd.Timedelta(seconds=(self.all_frames / self.fps))

    def set_sampling_strategy(self, sampling_strategy: VideoSampling):
        self.sampling_strategy = sampling_strategy

    def reset(self, batch_size_in_seconds=None, frame_position=0):
        self.overall_frame_counter = frame_position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start_position + frame_position)
        self.set_aligned_start_time()
        if batch_size_in_seconds is not None:
            self.set_batch_size(round(batch_size_in_seconds * self.fps))

    def set_end_time(self, other_end_time):
        if self.end_time > other_end_time:
            self.end_time = other_end_time
        self.all_frames = int((self.end_time - self._aligned_start_time).total_seconds() * self.fps)

    def extract(self, start_index=None, end_index=None):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start_position)

        self.df = self.extraction_method.setup_df()
        i = 0
        frame_nr = 0
        time_for_frame = self.start_time + dt.timedelta(
            microseconds=(1 / self.fps) * 1000000) * self.overall_frame_counter
        end = self.get_batch_size() if self.get_batch_size() < self.all_frames else self.all_frames
        while frame_nr < end:
            success, frame = self.cap.read()

            if not success or time_for_frame >= self.end_time:
                break
            i += 1
            if time_for_frame >= self._aligned_start_time:
                if self.sampling_strategy is not None:
                    frame = self.sampling_strategy.sample(frame)
                if frame is not False:
                    self.extraction_method.extract(frame, time_for_frame)
                frame_nr += 1

            time_for_frame += dt.timedelta(microseconds=(1 / self.fps) * 1000000)
            self.overall_frame_counter += 1

        self.time_stamps = pd.concat([self.time_stamps, pd.Series(self.df.index)], axis=0)

        return np.asarray(self.df[self.extraction_method.column_names].to_list())
