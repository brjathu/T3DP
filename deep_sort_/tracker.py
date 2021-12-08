"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import
import numpy as np
from . import linear_assignment
from .track import Track


class Tracker:

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx)
            
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
        
        return matches
        
    
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections = linear_assignment.matching_cascade(gated_metric, 
                                                                                                 self.metric.matching_threshold, 
                                                                                                 self.max_age,
                                                                                                 self.tracks, 
                                                                                                 detections, 
                                                                                                 confirmed_tracks)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, detection_id):
        self.tracks.append(Track(self._next_id, self.n_init, self.max_age, detection.feature, detection.tlwh, detection_id))
        self._next_id += 1
