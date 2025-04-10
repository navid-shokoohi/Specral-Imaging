import cv2
import numpy as np
import colour

# Load and normalize image
image_bgr = cv2.imread("./1.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

# Spectral shape: 380–780 nm in 5 nm steps
wavelengths = SpectralShape(380, 780, 5)
n_waves = len(wavelengths.range())
spectral_cube = np.zeros((height, width, n_waves))

# Get sRGB colourspace
srgb = colour.RGB_COLOURSPACES['sRGB']
illuminant = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# Loop over pixels
for i in range(height):
    for j in range(width):
        rgb = image_rgb[i, j]
        xyz = RGB_to_XYZ(rgb, srgb.whitepoint, illuminant, srgb.matrix_RGB_to_XYZ)
        sd = XYZ_to_sd_Meng2015(xyz)
        aligned_sd = sd.align(wavelengths)
        spectral_cube[i, j, :] = aligned_sd.values

print("Spectral cube shape:", spectral_cube.shape)
