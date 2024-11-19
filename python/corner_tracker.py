import numpy as np
import cv2
import os
from tqdm import tqdm


lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

feature_params = dict(maxCorners=150,
                      qualityLevel=0.01,
                      minDistance=30,
                      blockSize=3)

misc_params = dict(visualize=False,
                   detect_error=True,
                   tracking_error_threshold=0.3,
                   wait=1)


class CornerTracker:
    def __init__(self, image_paths, lk_params=lk_params, feature_params=feature_params, misc_params=misc_params):
        self.image_paths = image_paths
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.misc_params = misc_params
        self.prev_corners = None
        self.prev_frame = None
        self.index = 0
        self.feature_intensities = {} # {feature_id: np.array([intensity, ...])}
        self.feature_coords = {} # {feature_id: np.array([[x, y], ...])}
        self.active_features_per_frame = {i: [] for i in range(len(self.image_paths))} # {frame_id: [feature_id]}
        self.tracked_features_per_frame = {i: [] for i in range(len(self.image_paths))} # {frame_id: [feature_id]}
        self.error_frames = [] if misc_params['detect_error'] else None
        self.check_params()
    
    def check_params(self):
        assert self.feature_params['maxCorners'] > 0, 'maxCorners should be positive'
        assert self.feature_params['qualityLevel'] > 0, 'qualityLevel should be positive'
        assert self.feature_params['minDistance'] > 0, 'minDistance should be positive'
        assert self.feature_params['blockSize'] > 0, 'blockSize should be positive'
        assert self.lk_params['winSize'][0] > 0 and self.lk_params['winSize'][1] > 0, 'winSize should be positive'
        assert self.lk_params['maxLevel'] > 0, 'maxLevel should be positive'
        assert self.lk_params['criteria'][1] > 0, 'maxCount should be positive'
        assert self.lk_params['criteria'][2] > 0, 'epsilon should be positive'
        assert self.misc_params['visualize'] in [True, False], 'visualize should be boolean'
        assert self.misc_params['detect_error'] in [True, False], 'detect_error should be boolean'
        if self.misc_params['visualize']:
            assert self.misc_params['wait'] >= 0, 'wait should be non-negative'
        if self.misc_params['detect_error']:
            assert 0 < self.misc_params['tracking_error_threshold'] < 1, 'tracking_error_threshold should be in (0, 1)'


    def in_border(self, points, image_shape):
        '''
        return if points are in valid region of image (0: invalid, positive: valid)
        '''
        h, w = image_shape
        x, y = points.T
        return (x >= 0) & (x < w) & (y >= 0) & (y < h)

    def in_mask(self, points, mask):
        '''
        return if points are in valid region of mask (0: invalid, positive: valid)
        '''
        if mask is None:
            return True
        x, y = (points.T).astype(np.int32)
        return mask[y, x] > 0

    def initialize_features(self, id):
        if id not in self.feature_coords.keys():
            self.feature_coords[id] = np.ones((len(self.image_paths), 2), dtype=np.float32) * -1

    def latest_feature_id(self):
        return max(self.feature_coords.keys())

    def get_error_frames(self):
        return self.error_frames

    def track_once(self) -> bool:
        # read image
        image_track = cv2.imread(self.image_paths[self.index], cv2.IMREAD_GRAYSCALE)

        # data used for tracking
        cur_frame = image_track
        cur_mask = np.ones_like(cur_frame) * 255

        # initialize
        if self.prev_frame is None:
            self.prev_corners = cv2.goodFeaturesToTrack(cur_frame, self.feature_params['maxCorners'],
                                                       0.01, self.feature_params['minDistance'])
            for i, feature in enumerate(self.prev_corners.reshape(-1, 2)):
                self.initialize_features(i)
                self.feature_coords[i][self.index] = feature
                self.active_features_per_frame[self.index].append(i)
                self.tracked_features_per_frame[self.index].append(i)
            self.index += 1
            self.prev_frame = cur_frame
            return True
        
        # track corners with LK, and remove invalid corners
        cur_corners, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, cur_frame, self.prev_corners, None, **self.lk_params)
        prev_corners_rev, status_rev, err_rev = cv2.calcOpticalFlowPyrLK(cur_frame, self.prev_frame, cur_corners, None, **self.lk_params)
        d = abs(self.prev_corners - prev_corners_rev).reshape(-1, 2).max(-1)
        good = d < 1 # shape (N,)
        status = status & status_rev & good.reshape(-1, 1)
        for s, corner in zip (status, cur_corners):
            if s == 0:
                continue
            elif not self.in_border(corner, cur_frame.shape[:2]) or not self.in_mask(corner, cur_mask):
                s[0] = 0

        # for each tracked feature, update tracking information
        self.tracked_features_per_frame[self.index] = [id for id, s in zip(self.active_features_per_frame[self.index-1], status) if s[0] == 1]
        self.active_features_per_frame[self.index] = [id for id, s in zip(self.active_features_per_frame[self.index-1], status) if s[0] == 1]
        good_prev_corners = self.prev_corners[status == 1].reshape(-1, 1, 2)
        good_cur_corners = cur_corners[status == 1].reshape(-1, 1, 2)

        if misc_params['detect_error']:
            if len(good_cur_corners) < self.misc_params['tracking_error_threshold'] * len(self.prev_corners):
                print(f'Error detected at frame {self.index}')
                self.error_frames.append(self.index)
                if misc_params['visualize']:
                    cv2.imshow('Error', cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2BGR))
                    cv2.waitKey(0)

        # add new corners if needed
        tracked_num = len(good_cur_corners)
        if tracked_num < self.feature_params['maxCorners']:
            for feature in good_cur_corners.reshape(-1, 2):
                cv2.circle(cur_mask, tuple(feature.astype(np.int32)), self.feature_params['minDistance'], 0, -1)
            new_corners = cv2.goodFeaturesToTrack(cur_frame, self.feature_params['maxCorners'] - tracked_num, 
                                                 0.01, self.feature_params['minDistance'], mask=cur_mask)
            if new_corners is not None:
                good_cur_corners = np.concatenate([good_cur_corners, new_corners], axis=0)
                start_id = self.latest_feature_id() + 1
                for i, feature in enumerate(new_corners.reshape(-1, 2)):
                    self.active_features_per_frame[self.index].append(start_id + i)

        for id, corner in zip(self.active_features_per_frame[self.index], good_cur_corners):
            self.initialize_features(id)
            self.feature_coords[id][self.index] = corner

        # update data
        self.prev_corners = good_cur_corners
        self.prev_frame = cur_frame
        self.index += 1

        # draw (optional)
        if self.misc_params['visualize']:
            image_vis = cv2.cvtColor(image_track, cv2.COLOR_GRAY2BGR)
            for i, (new, old) in enumerate(zip(good_cur_corners, good_prev_corners)):
                a, b = new.ravel().astype(np.int32)
                c, d = old.ravel().astype(np.int32)
                cv2.circle(image_vis, (a, b), 3, (0, 255, 0), -1)
                cv2.line(image_vis, (a, b), (c, d), (0, 255, 0), 2)
            if len(good_cur_corners) > len(good_prev_corners):
                for i in range(len(new_corners)):
                    a, b = new_corners[i].ravel().astype(np.int32)
                    cv2.circle(image_vis, (a, b), 7, (0, 0, 255), 1)
                    cv2.circle(image_vis, (a, b), 3, (0, 0, 255), -1)
            cv2.putText(image_vis, f'num: {len(good_cur_corners)}, tracked: {tracked_num}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('image_vis', image_vis)
            if cv2.waitKey(self.misc_params['wait']) == ord('q'):
                return False
            
        return True

    
    def track(self):
        for i in tqdm(range(len(self.image_paths)), ncols=80):
            if not self.track_once():
                break
        cv2.destroyAllWindows()
        return self

if __name__ == '__main__':
    folder = 'YOUR_FOLDER_PATH'
    image_paths = [os.path.join(folder, image_path)
            for image_path in sorted(os.listdir(folder)) if image_path.endswith('.png')]

    tracker = CornerTracker(image_paths, lk_params, feature_params, misc_params).track()
    print(tracker.get_error_frames())
    
    images_error = [image_paths[i] for i in tracker.get_error_frames()]
    for image_path in images_error:
        image = cv2.imread(image_path)
        cv2.imshow('Error', image)
        cv2.waitKey(0)