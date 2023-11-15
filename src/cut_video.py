import cv2
import argparse

def cut_video(input_path, start_time, end_time, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get the frame rate of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the current frame number to the start frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Loop through the frames from the start frame number to the end frame number
    for i in range(start_frame, end_frame):
        # Read the current frame
        ret, frame = cap.read()

        # Write the current frame to the output video
        out.write(frame)

    # Release the input and output video objects
    cap.release()
    out.release()

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to the input video')
    parser.add_argument('start_time', type=float, help='start time in seconds')
    parser.add_argument('end_time', type=float, help='end time in seconds')
    parser.add_argument('output_path', help='path to the output video')
    args = parser.parse_args()

    # Call the cut_video function with the command line arguments
    cut_video(args.input_path, args.start_time, args.end_time, args.output_path)
