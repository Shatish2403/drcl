# preprocessing.py
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, clip_limit_l=5.0, clip_limit_ab=1.5, tile_grid_size=(8, 8), gamma_scale=1.2):
        self.clip_limit_l = clip_limit_l
        self.clip_limit_ab = clip_limit_ab
        self.tile_grid_size = tile_grid_size
        self.gamma_scale = gamma_scale
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_l = cv2.createCLAHE(clipLimit=self.clip_limit_l, tileGridSize=self.tile_grid_size)
        clahe_ab = cv2.createCLAHE(clipLimit=self.clip_limit_ab, tileGridSize=self.tile_grid_size)
        l_clahe = clahe_l.apply(l)
        a_clahe = clahe_ab.apply(a)
        b_clahe = clahe_ab.apply(b)
        lab_clahe = cv2.merge((l_clahe, a_clahe, b_clahe))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def adaptive_gamma_correction(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray) / 255.0
        gamma = np.clip(1.0 + (self.gamma_scale - 1.0) * (0.5 - mean_intensity), 0.5, 2.0) if mean_intensity > 0 else 1.0
        lookup_table = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, lookup_table)

    def preprocess(self, image):
        img_gamma = self.adaptive_gamma_correction(image)
        img_clahe = self.apply_clahe(img_gamma)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return self.transform(img_pil)
