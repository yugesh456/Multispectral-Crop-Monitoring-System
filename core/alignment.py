import cv2
import numpy as np

class AgriScanAligner:
    """
    Anti-Gravity Alignment Engine.
    Uses SIFT-based keypoint detection, FLANN/BF matching, and Homography mapping
    to align Blue, Green, Red, and Red-Edge bands with the reference NIR band.
    """
    def __init__(self, n_features=1000):
        # Initialize OpenCV SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        # BFMatcher with L2 norm for SIFT descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def align_stack(self, unaligned_stack):
        """
        Aligns a 5-band unaligned stack.
        unaligned_stack: numpy array of shape (5, H, W)
        Returns:
          aligned_stack: shape (5, H, W) perfectly aligned.
          homographies: list of 4 matrices mapping each band to the NIR grid.
          match_visualization_data: matching keypoints data for frontend display.
        """
        h, w = unaligned_stack.shape[1], unaligned_stack.shape[2]
        aligned_stack = np.zeros_like(unaligned_stack)
        
        # NIR (index 4) is our stable reference channel
        ref_band = (unaligned_stack[4] * 255.0).astype(np.uint8)
        aligned_stack[4] = unaligned_stack[4]

        # Extract features from the NIR reference band
        ref_kp, ref_des = self.sift.detectAndCompute(ref_band, None)
        
        homographies = []
        visualizations = []

        # Iterate over other bands: 0: Blue, 1: Green, 2: Red, 3: Red-Edge
        for band_idx in range(4):
            target_band = (unaligned_stack[band_idx] * 255.0).astype(np.uint8)
            
            # Extract keypoints and descriptors
            tar_kp, tar_des = self.sift.detectAndCompute(target_band, None)
            
            # Fallback if no features detected
            if ref_des is None or tar_des is None or len(ref_kp) < 4 or len(tar_kp) < 4:
                aligned_stack[band_idx] = unaligned_stack[band_idx]
                homographies.append(np.eye(3))
                visualizations.append([])
                continue
                
            # Match descriptors
            matches = self.matcher.match(tar_des, ref_des)
            # Sort by matching distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take top matches (up to 40)
            good_matches = matches[:40]
            
            if len(good_matches) >= 4:
                # Extract coordinates
                src_pts = np.float32([tar_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Compute Homography using RANSAC
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Warp target band
                    warped = cv2.warpPerspective(unaligned_stack[band_idx], H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                    aligned_stack[band_idx] = warped
                    homographies.append(H)
                    
                    # Package match lines for visualization in the UI
                    # (only take matched keypoints that RANSAC kept)
                    matches_mask = mask.ravel().tolist()
                    band_matches = []
                    for idx, m in enumerate(good_matches):
                        if idx < len(matches_mask) and matches_mask[idx]:
                            band_matches.append({
                                'src': {'x': tar_kp[m.queryIdx].pt[0], 'y': tar_kp[m.queryIdx].pt[1]},
                                'dst': {'x': ref_kp[m.trainIdx].pt[0], 'y': ref_kp[m.trainIdx].pt[1]}
                            })
                    visualizations.append(band_matches)
                else:
                    # Fallback on Homography calculation failure
                    aligned_stack[band_idx] = unaligned_stack[band_idx]
                    homographies.append(np.eye(3))
                    visualizations.append([])
            else:
                # Fallback if too few matches
                aligned_stack[band_idx] = unaligned_stack[band_idx]
                homographies.append(np.eye(3))
                visualizations.append([])

        return aligned_stack, homographies, visualizations
