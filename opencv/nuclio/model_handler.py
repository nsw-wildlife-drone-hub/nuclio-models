# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import cv2
from copy import copy
import numpy as np

class ModelHandler:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerMedianFlow_create()

    def infer(self, image, shape, state):
        image = np.array(image)
        state = np.array(state)
        
        x1, y1, x2, y2 = shape
        w = x1 - x2
        h = y1 - y2
        bbox = [x1, y1, w, h]
        
        # init tracking
        if state is None: 
            pass
            
        # track
        else:
            tracker = copy(self.tracker)
            ret = tracker.init(state, bbox)
            ret, bbox = tracker.update(image)
            
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        shape = [x1, y1, x2, y2]

        return shape
