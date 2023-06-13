# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import cv2
from copy import copy
import numpy as np

class ModelHandler:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerMedianFlow_create()
        self.state = None

    def infer(self, image, shape, state):
        image = np.array(image)
        
        # init tracking
        if state is None: 
            x1, y1, x2, y2 = shape
            w = x1 - x2
            h = y1 - y2
            bbox = [x1, y1, w, h]
            self.state = copy(self.tracker)
            self.state.init(image, bbox)
            
        # track
        else:
            _, bbox = self.state.update(image)
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            shape = [x1, y1, x2, y2]

        state = None
        return shape, state
