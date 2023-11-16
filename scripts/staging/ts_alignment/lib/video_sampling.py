import cv2


class VideoSampling:
    def __init__(self, data_loader):
        self.downscale = False
        self.rm_background = False
        self.down_sample = False
        self.fgbg = None
        self.scale = None
        self.down_sampling_rate = None
        self.mask = None
        self.data_loader = self.set_data_loader(data_loader)

    def set_data_loader(self, data_loader):
        data_loader.set_sampling_strategy(self)
        return data_loader

    def down_sample_video(self, sampling_rate_in_percent):
        self.down_sample = True
        self.down_sampling_rate = sampling_rate_in_percent

    def downscale_frames(self, scale_to_percentage):
        self.downscale = True
        self.scale = scale_to_percentage / 100.0

    def remove_background(self, history=500, distTh=500.0, detectShadows=True):
        self.rm_background = True
        self.fgbg = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=distTh,
                                                      detectShadows=detectShadows)

    def sample(self, frame):
        if self.down_sample:
            new_rate = self.data_loader.fps / (self.data_loader.fps * (1 - (self.down_sampling_rate / 100)))
            if self.data_loader.overall_frame_counter % new_rate >= 1:
                return False

        if self.downscale:
            frame = self._downscale_frame(frame)

        if self.rm_background:
            frame = self._subtract_background(frame)

        return frame

    def _subtract_background(self, frame):
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = self.fgbg.apply(cv2.blur(src_gray, (5, 5)))
        return cv2.bitwise_and(frame, frame, mask=self.mask)

    def _downscale_frame(self, frame):
        if self.data_loader.extraction_method.frame_needs_resizing:
            new_width = self.data_loader.x.frame_width
            new_height = self.data_loader.x.frame_height
            frame = cv2.resize(frame, (new_height, new_width), interpolation=cv2.INTER_AREA)

        return cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
