import cv2

import numpy as np
import cv2

# video cropping code taken from
# https://stackoverflow.com/a/61724147

# Open the video
cap = cv2.VideoCapture("videos/brach-naples.mp4")

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Here you can define your croping values
w = 512
h = 512
x = w_frame // 2 - w // 2 - 200
y = h_frame // 2 - h // 2

# output
out = cv2.VideoWriter(
    "videos/brach-naples-crop.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)


# Now we start
while cap.isOpened():
    ret, frame = cap.read()

    cnt += 1  # Counting frames

    # Avoid problems when video finish
    if ret == True:
        # Croping the frame
        crop_frame = frame[y : y + h, x : x + w]

        # Percentage
        xx = cnt * 100 / frames
        print(int(xx), "%")

        out.write(crop_frame)

        # Just to see the video in real time
        cv2.imshow("frame", frame)
        cv2.imshow("croped", crop_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
