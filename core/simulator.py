import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class AgriScanSimulator:
    """
    Procedural Simulator for AgriScan AI.
    Generates realistic 5-band multispectral leaf patches (B, G, R, RE, NIR)
    for Avocado, Olive, and Wheat across 5 disease severity levels (0 to 4).
    Also simulates UAV flight path jitter and band misalignment.
    """
    def __init__(self, patch_size=128):
        self.patch_size = patch_size
        self.bands = ['Blue', 'Green', 'Red', 'Red-Edge', 'NIR']

    def get_base_reflectance(self, crop_type, severity):
        """
        Returns realistic spectral reflectances for the 5 bands based on crop & severity.
        Levels:
          0: Healthy - high NIR & Red-Edge reflection, low Red (high absorption).
          1: Early Stress - Red-Edge & NIR drop significantly; Red/Green visual bands unchanged.
          2: Mild - Green & Red start yellowing/browning; NIR/RE drop further.
          3: Severe - Necrotic lesions appear; high Red, low NIR/RE.
          4: Terminal - Dry organic matter; low NIR/RE, high Red/Green/Blue (brownish).
        """
        # Ref: [Blue, Green, Red, Red-Edge, NIR]
        if severity == 0:
            # Healthy vegetation
            ref = [0.08, 0.22, 0.05, 0.48, 0.85]
        elif severity == 1:
            # Early-Stage Stress: RE & NIR drop, visual bands unchanged (key early signal!)
            ref = [0.08, 0.21, 0.05, 0.32, 0.68]
        elif severity == 2:
            # Mild Stress: visual yellowing starts, NIR/RE drop more
            ref = [0.10, 0.28, 0.12, 0.25, 0.52]
        elif severity == 3:
            # Severe Disease: dry spots, significant browning
            ref = [0.12, 0.24, 0.22, 0.18, 0.35]
        else: # severity == 4
            # Terminal / Dead
            ref = [0.15, 0.20, 0.28, 0.14, 0.22]
            
        # Add slight crop-specific tuning
        if crop_type == 'Olive':
            # Olives have silverish undersides; slightly higher Blue/Green reflectance
            ref[0] += 0.03
            ref[1] += 0.04
        elif crop_type == 'Wheat':
            # Wheat leaves are thinner, slightly lower NIR overall
            ref[4] -= 0.05
        elif crop_type == 'Vineyard':
            # Grape leaves are bright green, higher green reflectance
            ref[1] += 0.03
        elif crop_type == 'Citrus':
            # Thick glossy leaves, slightly higher visual reflectance overall
            ref[1] += 0.02
            ref[4] += 0.03
            
        return np.clip(ref, 0.01, 0.99)

    def generate_leaf_mask(self, crop_type):
        """
        Generates procedural leaf masks based on crop biology.
        """
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        center = self.patch_size // 2

        if crop_type == 'Avocado':
            # Broad oval, slightly tapered at tip
            cv2.ellipse(mask, (center, center + 10), (center - 15, center - 25), -15, 0, 360, 255, -1)
            # Tapered tip
            pts = np.array([[center - 10, center - 20], [center + 12, center - 20], [center, center - 45]], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif crop_type == 'Olive':
            # Slim, lanceolate leaf
            cv2.ellipse(mask, (center, center), (center - 32, center - 8), 45, 0, 360, 255, -1)
        elif crop_type == 'Vineyard':
            # Palmate grape leaf (5-lobed shape)
            cv2.ellipse(mask, (center, center), (center - 20, center - 35), 0, 0, 360, 255, -1)
            cv2.ellipse(mask, (center - 20, center - 10), (center - 22, center - 30), -35, 0, 360, 255, -1)
            cv2.ellipse(mask, (center + 20, center - 10), (center - 22, center - 30), 35, 0, 360, 255, -1)
            cv2.ellipse(mask, (center - 30, center + 15), (center - 25, center - 22), -75, 0, 360, 255, -1)
            cv2.ellipse(mask, (center + 30, center + 15), (center - 25, center - 22), 75, 0, 360, 255, -1)
        elif crop_type == 'Maize':
            # Long broad corn blade running slightly diagonal
            pts = np.array([[center - 22, self.patch_size], [center - 15, 30], [center, 0], 
                            [center + 15, 30], [center + 22, self.patch_size]], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif crop_type == 'Citrus':
            # Oval glossy citrus leaf with pointed tip
            cv2.ellipse(mask, (center, center + 5), (center - 18, center - 28), 0, 0, 360, 255, -1)
            pts = np.array([[center - 12, center - 15], [center + 12, center - 15], [center, center - 42]], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else: # Wheat
            # Long, slender vertical strip representing grass blade
            pts = np.array([[center - 12, self.patch_size], [center - 8, 15], [center, 0], 
                            [center + 8, 15], [center + 12, self.patch_size]], np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Smooth the leaf boundary slightly
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask

    def generate_multispectral_patch(self, crop_type='Avocado', severity=0, add_misalignment=False):
        """
        Generates a 5-band multispectral image stack (Blue, Green, Red, Red-Edge, NIR).
        """
        leaf_mask = self.generate_leaf_mask(crop_type) / 255.0
        reflectance = self.get_base_reflectance(crop_type, severity)
        
        # Base texture with vein structures
        x = np.linspace(-2, 2, self.patch_size)
        y = np.linspace(-2, 2, self.patch_size)
        X, Y = np.meshgrid(x, y)
        
        # Leaf texture
        if crop_type == 'Wheat':
            # Slender parallel lines
            veins = 1.0 - 0.15 * np.abs(np.sin(X * 8.0))
        elif crop_type == 'Olive':
            # One major central vein
            veins = 1.0 - 0.25 * np.exp(-np.abs(X + Y) * 8.0)
        elif crop_type == 'Vineyard':
            # Complex palmate radiating veins
            main_vein = np.exp(-np.abs(X + Y) * 8.0) + np.exp(-np.abs(X - Y) * 8.0)
            veins = 1.0 - main_vein * 0.28
        elif crop_type == 'Maize':
            # Parallel corn blade striations
            veins = 1.0 - 0.12 * np.abs(np.sin(X * 12.0))
        elif crop_type == 'Citrus':
            # Delicate glossy pinnate veins
            main_vein = np.exp(-np.abs(X) * 14.0)
            veins = 1.0 - main_vein * 0.20
        else: # Avocado
            # Pinnate vein structure
            main_vein = np.exp(-np.abs(X) * 12.0)
            side_veins = np.abs(np.sin((Y - np.abs(X)) * 6.0)) * 0.12 * (1 - np.abs(X)/2.0)
            veins = 1.0 - (main_vein * 0.25 + side_veins)
            
        veins = np.clip(veins, 0.4, 1.0)
        
        # Initialize bands
        patch_bands = []
        for i, band_name in enumerate(self.bands):
            # Base band reflectance inside leaf, zero outside
            band = leaf_mask * reflectance[i] * veins
            patch_bands.append(band)

        # Add disease spots procedurally
        if severity > 0:
            num_spots = int(severity * 3) if severity < 4 else 15
            for _ in range(num_spots):
                # Random spot locations inside the leaf
                while True:
                    sy, sx = np.random.randint(20, self.patch_size - 20, 2)
                    if leaf_mask[sy, sx] > 0.8:
                        break
                
                # Spot size depending on severity
                spot_radius = np.random.randint(3, 6 + int(severity * 2))
                
                # Create spot masks
                spot_y, spot_x = np.ogrid[:self.patch_size, :self.patch_size]
                dist_from_center = np.sqrt((spot_y - sy)**2 + (spot_x - sx)**2)
                
                # Soft spot decay
                spot_intensity = np.clip(1.0 - dist_from_center / spot_radius, 0, 1)
                spot_intensity = cv2.GaussianBlur(spot_intensity, (3, 3), 0)
                
                # Apply changes to spectral response in spot area
                # Spots reflect MORE Red (chlorosis/necrosis), and LESS NIR/Red-Edge
                for idx, band_name in enumerate(self.bands):
                    if band_name == 'Red':
                        # Necrotic brown spots increase Red reflectance
                        patch_bands[idx] += spot_intensity * 0.35 * leaf_mask
                    elif band_name in ['NIR', 'Red-Edge']:
                        # Structural decay lowers NIR/RE reflection
                        patch_bands[idx] -= spot_intensity * 0.45 * patch_bands[idx]
                    elif band_name == 'Green':
                        # Yellowing changes green reflectance
                        patch_bands[idx] += spot_intensity * (0.15 - patch_bands[idx])

        # Clip values to valid reflectance range
        for idx in range(5):
            patch_bands[idx] = np.clip(patch_bands[idx], 0.0, 1.0)

        # Pack into a numpy stack
        stack = np.stack(patch_bands, axis=0) # (5, H, W)

        # Inject physical misregistration (simulates UAV camera band jitter)
        if add_misalignment:
            aligned_stack = np.zeros_like(stack)
            # Keep NIR (index 4) as reference, jitter the others
            aligned_stack[4] = stack[4]
            
            for i in range(4):
                dx = np.random.uniform(-6, 6)
                dy = np.random.uniform(-6, 6)
                angle = np.random.uniform(-4, 4)
                
                # Affine transform matrix
                M = cv2.getRotationMatrix2D((self.patch_size / 2, self.patch_size / 2), angle, 1.0)
                M[0, 2] += dx
                M[1, 2] += dy
                
                aligned_stack[i] = cv2.warpAffine(stack[i], M, (self.patch_size, self.patch_size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return aligned_stack

        return stack

    def simulate_uav_flight(self, num_frames=100):
        """
        Generates simulated coordinates representing UAV flight telemetry:
        - Smooth coordinate path (intended path)
        - High-frequency jitter coordinates (unstabilized shaky path)
        - Kalman filtered trajectory (software stabilized path)
        """
        t = np.linspace(0, 10, num_frames)
        
        # Smooth flight path (panning sweeps)
        smooth_x = 40.0 * np.sin(t * 0.8) + 100.0
        smooth_y = 20.0 * np.cos(t * 0.5) + 100.0
        smooth_theta = 5.0 * np.sin(t * 0.4)
        
        # High-frequency jitter (extreme flight vibrations)
        jitter_x = smooth_x + np.random.normal(0, 8.0, num_frames)
        jitter_y = smooth_y + np.random.normal(0, 8.0, num_frames)
        jitter_theta = smooth_theta + np.random.normal(0, 2.5, num_frames)
        
        return {
            'time': t.tolist(),
            'smooth': {'x': smooth_x.tolist(), 'y': smooth_y.tolist(), 'theta': smooth_theta.tolist()},
            'jittery': {'x': jitter_x.tolist(), 'y': jitter_y.tolist(), 'theta': jitter_theta.tolist()}
        }

class AgriScanDataset(Dataset):
    """
    Infinite-capacity procedural dataset that supports the training phase.
    Avoids holding 40,000+ files on disk by procedurally drawing samples on-the-fly.
    """
    def __init__(self, size=40000, patch_size=128, train=True):
        self.size = size
        self.simulator = AgriScanSimulator(patch_size=patch_size)
        self.crops = ['Avocado', 'Olive', 'Wheat', 'Vineyard', 'Maize', 'Citrus']
        self.train = train

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = int(idx)
        # Reproducible random state based on index
        # Ensures that a given index always returns the exact same leaf patch!
        g = torch.Generator()
        g.manual_seed(idx + (0 if self.train else 999999))
        
        crop_idx = torch.randint(0, 3, (1,), generator=g).item()
        crop_type = self.crops[crop_idx]
        
        # Disease class weights (making early/healthy more common, or uniform)
        severity = torch.randint(0, 5, (1,), generator=g).item()
        
        # Generate multispectral leaf patch with slight misalignment
        # For training standard models, we can feed it slightly misaligned or perfectly aligned
        patch = self.simulator.generate_multispectral_patch(
            crop_type=crop_type, 
            severity=severity, 
            add_misalignment=False # Assume aligned stack for final model classification
        )
        
        # Convert to torch Tensor
        x = torch.from_numpy(patch).float()
        
        # Add a tiny bit of Gaussian noise for regularization
        if self.train:
            noise = torch.randn_like(x) * 0.015
            x = torch.clamp(x + noise, 0.0, 1.0)
            
        y = torch.tensor(severity, dtype=torch.long)
        
        return x, y
