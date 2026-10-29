import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os

os.makedirs('outputs/task2_features', exist_ok=True)

# ---------- Load Cube Images ----------
img1 = cv2.imread('data/Original_Photo.jpeg')
img2 = cv2.imread('data/Original_photo_02.jpeg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ---------- 1. HOG Visualization ----------
hog_features, hog_image = hog(gray1, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Cube Photo')
ax[0].axis('off')
ax[1].imshow(hog_image_rescaled, cmap='gray')
ax[1].set_title('HOG Features (shows cube edges/grid structure)')
ax[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/task2_features/hog_comparison.jpg', dpi=150)
plt.close()

print("HOG descriptor length:", len(hog_features))

# ---------- 2. SIFT Keypoints + Descriptors ----------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"Keypoints in Image 1: {len(kp1)}")
print(f"Keypoints in Image 2: {len(kp2)}")

img1_kp = cv2.drawKeypoints(img1, kp1, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('outputs/task2_features/sift_keypoints_cube1.jpg', img1_kp)

img2_kp = cv2.drawKeypoints(img2, kp2, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('outputs/task2_features/sift_keypoints_cube2.jpg', img2_kp)

# ---------- 3. Match Features (BFMatcher + Lowe's Ratio Test) ----------
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
ambiguous_matches = 0
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
    else:
        ambiguous_matches += 1   # likely due to repetitive cube squares

match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('outputs/task2_features/sift_matches_original.jpg', match_img)

print(f"\n--- Original Pair Matching ---")
print(f"Good matches: {len(good_matches)}")
print(f"Ambiguous/rejected matches (likely due to repetitive cube pattern): {ambiguous_matches}")

# ---------- 4. Repeat Matching After Rotation ----------
rows, cols = img2.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # rotate 45 degrees
img2_rotated = cv2.warpAffine(img2, M, (cols, rows))
gray2_rot = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)

kp2_rot, des2_rot = sift.detectAndCompute(gray2_rot, None)
matches_rot = bf.knnMatch(des1, des2_rot, k=2)

good_matches_rot = []
for m, n in matches_rot:
    if m.distance < 0.75 * n.distance:
        good_matches_rot.append(m)

match_img_rot = cv2.drawMatches(img1, kp1, img2_rotated, kp2_rot, good_matches_rot, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('outputs/task2_features/sift_matches_rotated.jpg', match_img_rot)

print(f"\n--- After 45° Rotation ---")
print(f"Good matches: {len(good_matches_rot)}")

# ---------- 5. Repeat Matching After Resizing ----------
img2_resized = cv2.resize(img2, None, fx=0.5, fy=0.5)
gray2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

kp2_res, des2_res = sift.detectAndCompute(gray2_resized, None)
matches_res = bf.knnMatch(des1, des2_res, k=2)

good_matches_res = []
for m, n in matches_res:
    if m.distance < 0.75 * n.distance:
        good_matches_res.append(m)

match_img_res = cv2.drawMatches(img1, kp1, img2_resized, kp2_res, good_matches_res, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('outputs/task2_features/sift_matches_resized.jpg', match_img_res)

print(f"\n--- After 50% Resize ---")
print(f"Good matches: {len(good_matches_res)}")

# ---------- Summary Table (for your report) ----------
print("\n=== SUMMARY TABLE ===")
print(f"{'Condition':<20}{'Good Matches':<15}")
print(f"{'Original pair':<20}{len(good_matches):<15}")
print(f"{'Rotated 45°':<20}{len(good_matches_rot):<15}")
print(f"{'Resized 50%':<20}{len(good_matches_res):<15}")

print("\nTask 2 completed. Check outputs/task2_features folder.")