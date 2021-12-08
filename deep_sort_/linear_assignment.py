"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

INFTY_COST = 1e+5

def greedy_assignment(a, max_distance):
    a = np.array(a)
    assignments = []
    for x_ in range(a.shape[0]):
        min_val = np.min(a)
        ids     = np.where(a==min_val)
        if(a[ids][0]<max_distance):
            assignments.append([ids[0][0], ids[1][0]])
        else:
            break
        a[ids[0],:] = 10**9
        a[:,ids[1]] = 10**9
    return a, np.array(assignments)
    
    
    
def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)    
    
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices

    matches, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices, unmatched_detections)


    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


