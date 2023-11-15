# !/usr/bin/env python3

## Script for converting from h264 to avi
# can be used by calling python in terminal and passing in the input and output paths
# e.g. python convert_h264.py /mnt/teams/TM_Lab/Arjun\ Bhaskaran/Social\ interaction\ project/Amir/videos/20231005093138_SI.h264 /mnt/teams/TM_Lab/Arjun\ Bhaskaran/Social\ interaction\ project/Amir/videos/test.avi
import cv2
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert h264 video to avi')
    parser.add_argument('input_path', type=str, help='path to input h264 video')
    parser.add_argument('output_path', type=str, help='path to output avi video')
    args = parser.parse_args()

    # Load the h264 video
    h264_video = cv2.VideoCapture(args.input_path)

    # Get the fps and resolution from the h264 video
    fps = h264_video.get(cv2.CAP_PROP_FPS)
    width = int(h264_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(h264_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi_video = cv2.VideoWriter(args.output_path, fourcc, 30, (width, height))

    total_frames = int(h264_video.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.tqdm(total=total_frames, unit="frame", unit_scale=True) as pbar:
        # Loop through the frames of the h264 video, convert each frame to avi, and write to output video
        while h264_video.isOpened():
            ret, frame = h264_video.read()
            if ret:
                avi_video.write(frame)
                pbar.update(1)
            else:
                break

    # Release everything
    h264_video.release()
    avi_video.release()
