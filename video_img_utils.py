import cv2
from deepposekit.io import video

import numpy as np
import cv2
import matplotlib.pyplot as plt


def cropvideo():
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


def video_side_by_side():
    cap1 = cv2.VideoCapture("videos/brach-2D-04-01-2022_dlc_gc4_epoch-10_thresh-07.mp4")
    cap2 = cv2.VideoCapture("videos/brach-2D-04-01-2022_dlc_gc4_epoch-50_thresh-07.mp4")
    cap3 = cv2.VideoCapture(
        "videos/brach-2D-04-01-2022_dlc_gc4_epoch-90_thresh-07_rgb.mp4"
    )

    # Some characteristics from the original video
    w_frame, h_frame = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps, frames = cap1.get(cv2.CAP_PROP_FPS), cap1.get(cv2.CAP_PROP_FRAME_COUNT)

    # Here you can define your croping values
    w = 512
    h = 480
    x = w_frame // 2 - w // 2
    y = h_frame // 2 - h // 2

    # output
    out = cv2.VideoWriter(
        "videos/brach-2D-04-01-2022-dlc_gc4_epochs.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w_frame * 3, h_frame),
    )

    cnt = 0

    # Now we start
    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        frame = np.hstack(
            (
                frame1,
                frame2,
                frame3,
            )
        )
        # frame = np.hstack(
        #     (
        #         frame1[y : y + h, x : x + w],
        #         frame2[y : y + h, x : x + w],
        #         frame3[y : y + h, x : x + w],
        #     )
        # )

        out.write(frame)

        cnt += 1  # Counting frames

        # Avoid problems when video finish
        if ret1 == True or ret2 == True or ret3 == True:

            # Just to see the video in real time
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_side_by_side()
