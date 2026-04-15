from systemds.scuro.representations.alignment import Alignment
import cv2 as cv
import numpy as np

class PHashAlignment(Alignment):
    def __init__(self):
        super().__init__("pHashAlignment")
        self.hasher = cv.img_hash.PHash_create()
        
    def compute_descriptor(self, segment):
        if segment.ndim == 3:
            return [self.hasher.compute(segment)]
        if segment.ndim == 4: # For videos
            descriptors = []
            for s in segment:
                frame = (s * 255).astype(np.uint8, copy=True)
                descriptors.append(self.hasher.compute(frame))
            return descriptors
        raise("PHashAlignment is only implemented for ndim=3 or ndim=4")
        
    def compare(self, a, b):
        return self.hasher.compare(a, b)