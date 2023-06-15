# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io
import base64
import cv2
import numpy as np

from PIL import Image

class ModelHandler:
    def __init__(self):
        self.tracker = cv2.legacy.TrackerMedianFlow_create()

    def infer(self, image, shape, state):
        # init tracking
        if state is None: 
            pass
            
        # track
        else:
            state = io.BytesIO(base64.b64decode(state))
            prev_image = Image.open(state)
            state = np.array(state)
            prev_image = np.array(prev_image)
            
            x1, y1, x2, y2 = shape
            w = x1 - x2
            h = y1 - y2
            bbox = [x1, y1, w, h]
            tracker = self.tracker
            
            ret = tracker.init(prev_image, bbox)
            ret, bbox = tracker.update(image)
            
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            shape = [x1, y1, x2, y2]

        return shape
