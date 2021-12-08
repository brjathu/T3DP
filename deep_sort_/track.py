"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np

def update_feature(a, b, th=0.5):
    
    a = np.array(a)
    b = np.array(b)
    a = np.reshape(a, (1, 512))
    b = np.reshape(b, (1, 512))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = a / np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.dot(a, b.T)
    print(dot)
    if(dot>=th):
        
        a = dot*a + (1-dot)*b
    return list(a[0])
    
    
    
class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted   = 3


class Track:
    def __init__(self, track_id, n_init, max_age, feature=None, bbox=None, detection_id=None):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Confirmed
        self.features = []
        self.mean_features = []
        self.all_features = []
        self.bbox = []
        self.detection_id = []
        if feature is not None:
            self.features.append(feature)
            self.mean_features.append(feature)
            self.all_features.append(feature)
            self.bbox.append(bbox)
            self.detection_id.append(detection_id)

        self._n_init = n_init
        self._max_age = max_age

    def predict(self):
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, detection_id):
        if(len(self.features)<200):
            self.features.append(detection.feature)
            self.all_features.append(detection.feature)
            self.bbox.append(detection.tlwh)
        else:
            self.features.append(detection.feature)
            self.features = self.features[1:]
            self.all_features.append(detection.feature)
            self.all_features = self.all_features[1:]
            self.bbox.append(detection.tlwh)
            self.bbox = self.bbox[1:]

        self.detection_id.append(detection_id)
                
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
