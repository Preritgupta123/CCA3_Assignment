import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------- Load Image ----------
img = cv2.imread('data/Original_Photo.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------- A. ADD NOISE ----------
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

gaussian_noisy = add_gaussian_noise(img)
sp_noisy = add_salt_pepper_noise(img)

# ---------- B. RESTORE (Filtering) ----------
gaussian_restored = cv2.GaussianBlur(gaussian_noisy, (5,5), 0)
median_restored = cv2.medianBlur(sp_noisy, 5)

# ---------- C. METRICS ----------
print("PSNR (Gaussian noisy vs original):", psnr(img, gaussian_noisy))
print("PSNR (Gaussian restored vs original):", psnr(img, gaussian_restored))
print("SSIM (Gaussian restored vs original):", ssim(img, gaussian_restored, channel_axis=2))

print("PSNR (S&P noisy vs original):", psnr(img, sp_noisy))
print("PSNR (Median restored vs original):", psnr(img, median_restored))
print("SSIM (Median restored vs original):", ssim(img, median_restored, channel_axis=2))

# ---------- Save outputs ----------
cv2.imwrite('outputs/task1_restoration/gaussian_noisy.jpg', cv2.cvtColor(gaussian_noisy, cv2.COLOR_RGB2BGR))
cv2.imwrite('outputs/task1_restoration/sp_noisy.jpg', cv2.cvtColor(sp_noisy, cv2.COLOR_RGB2BGR))
cv2.imwrite('outputs/task1_restoration/gaussian_restored.jpg', cv2.cvtColor(gaussian_restored, cv2.COLOR_RGB2BGR))
cv2.imwrite('outputs/task1_restoration/median_restored.jpg', cv2.cvtColor(median_restored, cv2.COLOR_RGB2BGR))

# ---------- D. MORPHOLOGICAL PROCESSING & SEGMENTATION ----------
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('outputs/task1_restoration/binary.jpg', binary)
cv2.imwrite('outputs/task1_restoration/erosion.jpg', erosion)
cv2.imwrite('outputs/task1_restoration/dilation.jpg', dilation)
cv2.imwrite('outputs/task1_restoration/opening.jpg', opening)
cv2.imwrite('outputs/task1_restoration/closing.jpg', closing)

# ---------- E. CANNY EDGE DETECTION ----------
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite('outputs/task1_restoration/canny_edges.jpg', edges)

# ---------- F. WATERSHED SEGMENTATION ----------
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray_ws = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_ws, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3), np.uint8)
opening_ws = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background
sure_bg = cv2.dilate(opening_ws, kernel, iterations=3)

# sure foreground
dist_transform = cv2.distanceTransform(opening_ws, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img_bgr, markers)
img_bgr[markers == -1] = [0, 0, 255]  # mark boundaries in red

cv2.imwrite('outputs/task1_restoration/watershed_result.jpg', img_bgr)

print("Task 1 completed. Check outputs/task1_restoration folder.")