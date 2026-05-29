import numpy as np
import cv2

class AgriScanSpectralEngine:
    """
    Vegetation Index Mapping Engine.
    Computes high-resolution NDVI and NDRE maps, colorizes them using premium palettes,
    and isolates stressed vegetative zones.
    """
    def __init__(self):
        pass

    def compute_ndvi(self, aligned_stack):
        """
        Computes NDVI = (NIR - Red) / (NIR + Red)
        aligned_stack: numpy array of shape (5, H, W).
          Index 2 is Red, Index 4 is NIR.
        """
        red = aligned_stack[2]
        nir = aligned_stack[4]
        
        denominator = nir + red
        # Prevent division by zero
        denominator = np.where(denominator == 0, 1e-6, denominator)
        
        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1.0, 1.0)

    def compute_ndre(self, aligned_stack):
        """
        Computes NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        aligned_stack: numpy array of shape (5, H, W).
          Index 3 is RedEdge, Index 4 is NIR.
        """
        red_edge = aligned_stack[3]
        nir = aligned_stack[4]
        
        denominator = nir + red_edge
        denominator = np.where(denominator == 0, 1e-6, denominator)
        
        ndre = (nir - red_edge) / denominator
        return np.clip(ndre, -1.0, 1.0)

    def colorize_index(self, index_map, cmap_type='healthy'):
        """
        Maps a 1-channel index [-1, 1] to a gorgeous, premium RGB color palette.
        - Healthy: Deep emerald green.
        - Stressed: Amber / Yellow / Orange.
        - Barren/Soil: Crimson / Earthy Brown.
        """
        h, w = index_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Scale index from [-1, 1] to [0, 1]
        scaled = (index_map + 1.0) / 2.0

        for r in range(h):
            for c in range(w):
                val = index_map[r, c]
                
                if val < 0.15:
                    # Soil, wood, stone: earthy brown/grey (RGB)
                    # Interpolate from grey (0, 0) to brown (0.15)
                    t = max(0, val + 1.0) / 1.15
                    color_map[r, c] = [
                        int(110 * t + 80 * (1-t)),  # Red
                        int(85 * t + 80 * (1-t)),   # Green
                        int(60 * t + 80 * (1-t))    # Blue
                    ]
                elif val < 0.45:
                    # Stressed vegetation / low biomass: amber to yellow
                    t = (val - 0.15) / 0.30
                    color_map[r, c] = [
                        230,                      # Red (warm amber)
                        int(140 + 90 * t),        # Green (yellowing)
                        int(30 * (1-t))           # Blue (vibrant)
                    ]
                elif val < 0.70:
                    # Moderately healthy: yellow-green to light emerald green
                    t = (val - 0.45) / 0.25
                    color_map[r, c] = [
                        int(230 * (1-t) + 16 * t), # Red
                        int(230 * (1-t) + 185 * t),# Green
                        int(30 * (1-t) + 129 * t)  # Blue
                    ]
                else:
                    # Ultra-healthy high biomass: deep emerald/teal
                    t = min(1.0, (val - 0.70) / 0.30)
                    color_map[r, c] = [
                        int(16 * (1-t) + 4 * t),   # Red
                        int(185 * (1-t) + 120 * t),# Green
                        int(129 * (1-t) + 90 * t)  # Blue
                    ]

        return color_map

    def isolate_stress_zones(self, index_map, min_threshold=0.18, max_threshold=0.48):
        """
        Identifies and isolates stressed vegetation pixels.
        Returns a binary mask where 255 denotes stressed zones.
        """
        stress_mask = np.logical_and(index_map >= min_threshold, index_map <= max_threshold)
        return (stress_mask * 255).astype(np.uint8)

    def create_rgb_composite(self, aligned_stack):
        """
        Creates an RGB false-color/true-color composite.
        We map Index 2 (Red), Index 1 (Green), and Index 0 (Blue) to standard RGB.
        """
        # Read channels
        r = aligned_stack[2]
        g = aligned_stack[1]
        b = aligned_stack[0]

        # Merge and scale to [0, 255]
        composite = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
        composite[..., 0] = np.clip(r * 255.0, 0, 255)
        composite[..., 1] = np.clip(g * 255.0, 0, 255)
        composite[..., 2] = np.clip(b * 255.0, 0, 255)
        
        return composite
        
    def create_cir_composite(self, aligned_stack):
        """
        Creates a Color Infrared (CIR) composite (NIR, Red, Green as RGB).
        Extremely useful for visual crop stress checking.
        """
        nir = aligned_stack[4]
        red = aligned_stack[2]
        green = aligned_stack[1]
        
        composite = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
        composite[..., 0] = np.clip(nir * 255.0, 0, 255) # Red channel gets NIR
        composite[..., 1] = np.clip(red * 255.0, 0, 255) # Green channel gets Red
        composite[..., 2] = np.clip(green * 255.0, 0, 255) # Blue channel gets Green
        
        return composite
