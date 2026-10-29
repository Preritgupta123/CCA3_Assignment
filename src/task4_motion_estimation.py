import cv2
import numpy as np

cap = cv2.VideoCapture('data/original_video.mp4')

# ---------- 1. Extract two consecutive frames ----------
ret, frame1 = cap.read()
ret, frame2 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# ---------- 2. Frame Differencing ----------
diff = cv2.absdiff(gray1, gray2)
_, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
cv2.imwrite('outputs/task4_motion/frame_differencing.jpg', diff_thresh)

# ---------- 3. GMM Background Subtraction ----------
cap2 = cv2.VideoCapture('data/Original_video.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

frame_count = 0
fgmask = None
while True:
    ret, frame = cap2.read()
    if not ret:
        break
    fgmask = backSub.apply(frame)
    frame_count += 1
    if frame_count == 30:   # save a mid-video sample mask
        cv2.imwrite('outputs/task4_motion/gmm_foreground_mask.jpg', fgmask)
cap2.release()

# ---------- 4. Lucas-Kanade Optical Flow ----------
cap3 = cv2.VideoCapture('data/original_video.mp4')
ret, old_frame = cap3.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

ret, frame = cap3.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

good_new = p1[st==1]
good_old = p0[st==1]

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
    frame = cv2.circle(frame, (int(a),int(b)), 5, (0,0,255), -1)

output_flow = cv2.add(frame, mask)
cv2.imwrite('outputs/task4_motion/lucas_kanade_optical_flow.jpg', output_flow)

cap3.release()
cap.release()

print("Task 4 completed. Check outputs/task4_motion folder.")