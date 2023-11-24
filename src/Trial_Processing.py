
# Import libraries
import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import cv2
import tdt
from matplotlib.animation import FuncAnimation
sys.path.append(os.path.abspath('../'))
import config

## Outline of Processing pipeline
# 1. Find corresponding video, csv, and photometry files base done the date in the title of the file
#   grouping together all the files with the closest time of recoding to eachother,
#  and then saving that grouping into a CSV indexed by the date
# 2. Load in the photometry data and the video data for each trial, then align their start times to the first cue onset,
#  using the title of the video as the start time and the second rising edge of the photometry signal as the cue onset
# 3. Segment the video and photometry data into trials based on the cue onset and the trial duration
# 4. Save the segmented photometry data by trial into a CSV file for each day
# 5. Save the segmented video data by trial into a mp4 file for each day

## Global Variables
# Regex's determin the format of the file names, note if the file name format changes then you will also need to update how the names and dates are extracted if they are formatted differently
NAME_REGEX = ".*_?[L | R]?_beambreak_time in s"
PHOTOMETRY_FILE_REGEX = "Social_Interaction-[0-9]*"
PHOTOMETRY_NAME_REGEX = ".*_F1_.*F2"
CSV_FILE_REGEX = "[0-9]*_log.csv"
VIDEO_FILE_REGEX = "[0-9]*_SI.h264"

# Date format parser
DATE_FORMAT = '%Y%m%d%H%M%S'
PHOTOMETRY_DATE_FORMAT = '%y%m%d'

## Function Definitions

## Parse Date from preferred format
# Information on datetime formats: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
# @param: date - string containing the date in the format of 'YYMMDDHHMMSS' followed by underscores
# @return: date - datetime object containing the date 
def parse_date(date, format=DATE_FORMAT):
    date = date.split('_')[0][:14]
    date = datetime.datetime.strptime(date, format)
    return date

## Get the names of the mice from the CSV file
# @param: CSV - pandas dataframe containing the CSV file
# @return: names - list of strings containing the names of the mice
def get_names(CSV):
    names = []
    for column in CSV.columns:
        if re.match(NAME_REGEX, column):
            name = column.split('_')[0]
            names.append(name)
    
    assert len(names) == 2, "Error: CSV file does not contain two mice"
    assert names[0] != names[1], "Error: CSV file contains two mice with the same name"
    assert names != [], "Error: CSV file does not contain any mice"

    return names

## Get the closest file to a given date
# @param: date - datetime object containing the date to match
# @param: files - list of strings containing the names of the files to match to the date
# @param: max - int containing the maximum number of seconds between the date and the file
# @return: closest_file - string containing the name of the file with the closest date to the given date
def get_closest_file(date, files, max):
    closest_match = None
    for file in files:
        file_date = parse_date(file)
        if abs((file_date - date).total_seconds()) <= max:
            closest_match = file
            max = abs((file_date - date).total_seconds())
    return closest_match

## Given a stream of data, find all the rising edges of the data above a threshold
## First lowpasses the data with a moving average filter, then takes the derivative of the data
## Then finds all the rising edges of the data above a threshold
# @param: data - numpy array containing the data
# @param: threshold - float containing the threshold to detect the rising edges
# @param: window - int containing the size of the moving average filter to use, use smaller values for lower samplerate
# @return: triggered_data - list of each separate block of data above the threshold
# @return: rising_edges - list of ints containing the indices of the rising edges of the data,
# @return: falling_edges - list of ints containing the indices of the falling edges of the data
def detect_triggers(data, threshold=0.5, window=1000):
    smoothed_data = np.convolve(data, np.ones(window)/window, mode='same')
    diff_data = np.diff(smoothed_data)
    diff_data = np.convolve(diff_data, np.ones(window)/window, mode='same')
    recording = False
    triggered_data = []
    rising_edges = []
    falling_edges = []
    for i in range(len(diff_data)):
        if diff_data[i] > threshold and not recording:
            recording = True
            # Add a offset of half the window to the rising edge to compensate for phase shift from the moving average filter
            rising_edges.append(i + window // 2)
            triggered_data.append([data[i]])
        elif diff_data[i] < -threshold and recording:
            recording = False
            falling_edges.append(i + window // 2)
            
        elif recording and diff_data[i] < threshold and diff_data[i] > -threshold:
            triggered_data[-1].append(data[i])
    return triggered_data, rising_edges, falling_edges


## Downsample a stream of data to a given sample rate from a given sample rate, by taking every Nth sample, does not lowpass the data
## could result in aliasing if there is high frequency noise in the data with frequency > target_fs/2
# @param: data - numpy array containing the data
# @param: fs - int containing the sample rate of the data
# @param: targer_fs - int containing the sample rate to downsample to
# @return: downsampled_data - numpy array containing the downsampled data
def downsample(data, fs, target_fs):
    assert fs >= target_fs, "Error: Target sample rate must be less than the sample rate of the data"
    if fs == target_fs:
        return data
    else:
        downsampled_data = []
        for i in range(0, len(data), int(fs/target_fs)):
            downsampled_data.append(data[i])
        return np.array(downsampled_data)

## Load the Photometry Streams using the TDT data from the path, and then split them up by the fiber activations and the photometry channels
# @param: path - string containing the path to the TDT data
# @param: sample_rate - float containing the sample rate of the photometry data, by default 1017 (fs of the TDT system)
# @param: trial_duration - int containing the length of the trial in seconds, by default 30, this should be how long the light was turned on for in the trial
# @param: skip_trials - int containing the number of trials to skip at the beginning of the data, by default 1, this is to skip the first trial which is usually before the experiment starts
# @param: threshold - float containing the threshold to detect the rising edges of the data, by default 10
# @param: csv_start_date - a datetime object with the cuetime of the first trial, if given it's used to determine how many blocks of data to skip at the start, if it is used then skip_trials is ignored
# @return: 405A, 405B, 465A, 465B - N x L numpy arrays containing the data from each channel, where N is the number of trials and L is the length of the trial
def load_photometry_streams(path, sample_rate=None, trial_duration=30, skip_trials=1, threshold=10, csv_start_date=None):
    # Load the TDT data from the path (this may take a while)
    data = tdt.read_block(path)

    if sample_rate is None:
        sample_rate = data.streams['_405A'].fs

    # Get the data from each channel, downsample it to samplerate if needed
    stream_405A = downsample(data.streams['_405A'].data, data.streams['_405A'].fs, sample_rate)
    stream_405C = downsample(data.streams['_405C'].data, data.streams['_405C'].fs, sample_rate)
    stream_465A = downsample(data.streams['_465A'].data, data.streams['_465A'].fs, sample_rate)
    stream_465C = downsample(data.streams['_465C'].data, data.streams['_465C'].fs, sample_rate)
    
    # get the start time of the photometry recording, and then create an array of timestamps for each sample
    photometry_start_date = data.info.start_date
    if csv_start_date is not None:
        offset_time = (csv_start_date - photometry_start_date).total_seconds()
        offset_samples = int(offset_time * sample_rate)
        print('Offsetting photometry data by ' + str(offset_samples) + ' samples')
        print(len(stream_405A))
        stream_405A = stream_405A[offset_samples:]
        stream_405C = stream_405C[offset_samples:]
        stream_465A = stream_465A[offset_samples:]
        stream_465C = stream_465C[offset_samples:]
        skip_trials = 1

    timestamps = np.linspace(0, len(stream_405A) / sample_rate, len(stream_405A))

    # All streams must be the same length
    assert len(stream_405A) == len(stream_405C) == len(stream_465A) == len(stream_465C), "Error: Photometry streams are not the same length"

    # Sum the data together from each channel and then detect the rising edges of the data
    total_stream_405 = stream_405A + stream_405C + stream_465A + stream_465C
    _ , start_indices, end_indices = detect_triggers(total_stream_405, threshold=threshold, window=int(sample_rate))

    print("Number of trials: " + str(len(start_indices)))

    # Split the data into trials based on the rising edges and the trial duration
    # Skips the Last trial because it is usually cut off
    try:
        trials_405A = ([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_405A, start_indices)[skip_trials:-1]])
        trials_405C = ([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_405C, start_indices)[skip_trials:-1]])
        trials_465A = ([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_465A, start_indices)[skip_trials:-1]])
        trials_465C = ([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_465C, start_indices)[skip_trials:-1]])
        trial_timestamps = ([ trial[:int(trial_duration*sample_rate)] for trial in np.split(timestamps, start_indices)[skip_trials:-1]])
        return trials_405A, trials_405C, trials_465A, trials_465C, trial_timestamps

    except IndexError:
        print("Error: Photometry data at path: " + path + " is shorter than the given trial duration, either trial duration is too long or the threshold is incorrect")
        return None, None, None, None
    
## Takes a CSV, Video, and photometry path, and writes the photometry data to a CSV file with each trial labelled, as well as video timestamps for each entry
## It first loads the photometry data then aligns the first trial to the first cue in the data, then loads the video data and finds the offset to the first cue and the frane rate
## Then it creates a CSV with a column for each photometry trace, as well as a column for the video timestamps and a collumn labelling each trial
# @param: CSV_path - string containing the path to the CSV file
# @param: Video_path - string containing the path to the video file
# @param: Photometry_path - string containing the path to the photometry file
# @param: output_path - string containing the path to the folder to save the CSV file to
# @param: F1_name - string containing the name of the first mouse
# @param: F2_name - string containing the name of the second mouse
# @param: trial_duration - int containing the length of the trial in seconds, by default 30, this should be how long the light was turned on for in the trial
# @param: skip_trials - int containing the number of trials to skip at the beginning of the data, by default 1, this is to skip the first trial which is usually before the experiment starts
# @param: sample_rate - float containing the sample rate of the photometry data, by default 1017 (fs of the TDT system)
# @param: cue_offset_seconds - the signed offset between the cue and the trigger sent to the photmetry data in seconds, e.g. if the trigger was 5 seconds before the cue it would be -5
# @param: use_time_alignment - bool, if true then the it will use the start datetimes of the CSV file and the video file to align the data, if false then it will use the skip trials parameter to align the data
# @return - df - pandas dataframe containing the photometry data, with each trial labelled, as well as video timestamps for each entry
def write_photometry_to_csv(CSV_path, Video_path, Photometry_path, output_path, F1_name, F2_name, trial_duration=30, skip_trials=1, sample_rate=None, cue_offset_seconds=0, use_time_alignment=False):
    experiment_csv = pd.read_csv(CSV_path)
    if use_time_alignment:
        experiment_start_date = parse_date(experiment_csv['cue_time in s'][0])
    else:
        experiment_start_date = None
    
    trials_405A, trials_405C, trials_465A, trials_465C, timestamps = load_photometry_streams(Photometry_path, sample_rate=sample_rate, trial_duration=trial_duration, skip_trials=skip_trials, csv_start_date=experiment_start_date)
    if sample_rate is None:
        sample_rate = 1017

    video = cv2.VideoCapture(Video_path)
    ## Video fps hardcapped 30 as cv2 doesn't always return the correct fps
    video_fps = 30
    video_start_time = parse_date(Video_path.split('/')[-1].split('_')[0])
    first_cue_onset = parse_date(experiment_csv['cue_time in s'][0])
    cue_delta_time_seconds = experiment_csv['cue_time in s'].apply(lambda x: (parse_date(x) - first_cue_onset).total_seconds())
    video_offset_seconds = (first_cue_onset - video_start_time).total_seconds() + cue_offset_seconds
    video_offset_frames = int(video_offset_seconds * video_fps)
    trial_length = int(trial_duration * sample_rate)
    video_length = int(trial_duration * video_fps)
    data = []
    print(len(timestamps))
    ## Cut off the first two directories of each path as they are the remote path
    CSV_path = '/'.join(CSV_path.split('/')[-2:])
    Video_path = '/'.join(Video_path.split('/')[-2:])
    Photometry_path = '/'.join(Photometry_path.split('/')[-2:])

    for i in range(len(cue_delta_time_seconds)):

        trial_405A = trials_405A[i]
        trial_405C = trials_405C[i]
        trial_465A = trials_465A[i]
        trial_465C = trials_465C[i]
        trial_timestamps = timestamps[i]
        video_frames = np.linspace(video_offset_frames + int(cue_delta_time_seconds[i] * video_fps), video_offset_frames + int(cue_delta_time_seconds[i] * video_fps) + video_length, trial_length)
        # round to nearest frame
        video_frames = np.round(video_frames)
        for j in range(trial_length):
            if i == 0 and j == 0:
                data.append({'Trial': i, '405A': trial_405A[j], '405C': trial_405C[j], '465A': trial_465A[j], '465C': trial_465C[j], 'Video Frame': video_frames[j], 'Timestamp': trial_timestamps[j], 'Sample Rate': sample_rate, 'Video FPS': video_fps, "Video Path": Video_path, 'Photometry Path': Photometry_path, 'CSV Path': CSV_path, 'F1 Name' : F1_name, 'F2 Name': F2_name})
            else:
                data.append({'Trial': i, '405A': trial_405A[j], '405C': trial_405C[j], '465A': trial_465A[j], '465C': trial_465C[j], 'Video Frame': video_frames[j], 'Timestamp': trial_timestamps[j], 'Sample Rate': None, 'Video FPS': None, "Video Path": None, 'Photometry Path': None, 'CSV Path': None, 'F1 Name' : None, 'F2 Name': None})

    df = pd.DataFrame(data, columns=['Trial', '405A', '405C', '465A', '465C', 'Video Frame', 'Timestamp', 'Sample Rate', 'Video FPS', "Video Path", 'Photometry Path', 'CSV Path', 'F1 Name', 'F2 Name'])
    df.to_csv(output_path, index=False)
    return df
        

## Find Matching CSV, Video, and Photometry Files
# @param: CSV_path - path to the folder containing the CSV files
# @param: Video_path - path to the folder containing the video files
# @param: Photometry_path - path to the folder containing the photometry files
# @param: output_path - path to the folder to save the CSV file containing the paths to the CSV, video, and photometry files, if not provided then the CSV file will not be saved
# @return: df - dataframe containing the paths to the CSV, video, and photometry files

def match_datafiles_by_date(CSV_path, Video_path, Photometry_path, output_path=None):
    # Get list of all files in each folder, filtering by regex
    CSV_files = [csv for csv in os.listdir(CSV_path) if re.match(CSV_FILE_REGEX, csv)]
    Video_files = [video for video in os.listdir(Video_path) if re.match(VIDEO_FILE_REGEX, video)]
    Photometry_files = [photometry for photometry in os.listdir(Photometry_path) if re.match(PHOTOMETRY_FILE_REGEX, photometry)]
    # Create dataframe to store the paths to the CSV, video, and photometry files
    df = pd.DataFrame(columns=['Date', 'F1 Name', 'F2 Name','CSV', 'Video', 'Photometry'])
    csv_video_pairs = dict()
    csv_photo_pairs = dict.fromkeys(CSV_files, None)
    # Pair together matching videos and CSV files based on the date in the title finding the closest match in time to eachother
    # rejecting any video with no CSV file withing a set time period of it
    # Append to dict of video files keyed by the CSV file
    for csv in CSV_files:
        # Get the date of the video file
        csv_date = parse_date(csv)
        # Get the path to the CSV file with the closest date to the video file
        closest_Video = get_closest_file(csv_date, Video_files, 60)
        if closest_Video is None:
            # If no CSV file was found, skip this video file and print a warning
            print("Warning: No Video found for this CSV file: " + csv)
            csv_video_pairs[csv] = None

        else:
            # Add the video file to the dict of video files keyed by the CSV file
            assert csv not in csv_video_pairs, "Error: CSV file already has a video file paired with it"
            csv_video_pairs[csv] = closest_Video

    # Iterate through all photometry files and pair them with the closest CSV file
    for photometry in Photometry_files:
        photometry_dir_date = parse_date(photometry.split('-')[-1], format=PHOTOMETRY_DATE_FORMAT)
        photometry_subdir = Photometry_path + photometry + '/'
        photometry_trials_by_name = [ name for name in os.listdir(photometry_subdir) if re.match(PHOTOMETRY_NAME_REGEX, name)]
        # Iterate through all CSV files and find each one that lands on the same day
        for csv_file in csv_video_pairs.keys():
            csv_date = parse_date(csv_file)
            if csv_date.date() == photometry_dir_date.date():
                csv_full_path = CSV_path + csv_file
                df_csv = pd.read_csv(csv_full_path)
                try:
                    names = get_names(df_csv)
                except AssertionError:
                    print("Error: CSV file does not contain two mice, or the mice have the same name")
                    print(csv_file)
                
                # Check if the names of the mice match the names of the photometry files
                for photometry_trial_of_the_day in photometry_trials_by_name:
                    photo_trial_name_list = photometry_trial_of_the_day.split('_')
                    if names[0] in photo_trial_name_list and names[1] in photo_trial_name_list:
                        # If they match, add the CSV file to the list of paired CSV files
                        matched_photometr_path = photometry_subdir + photometry_trial_of_the_day
                        csv_photo_pairs[csv_file] = matched_photometr_path
    
    # Iterate through all CSV files and append them to the dataframe with the paths to the CSV, video, and photometry files, date, and names
    for csv_file in csv_video_pairs.keys():
        csv_full_path = CSV_path + csv_file
        df_csv = pd.read_csv(csv_full_path)            
        if csv_video_pairs[csv_file] is not None:
            video_full_path = Video_path + csv_video_pairs[csv_file]
        else:
            video_full_path = None
        if csv_photo_pairs[csv_file] is not None:
            photometry_full_path = csv_photo_pairs[csv_file]
            F1_name = csv_photo_pairs[csv_file].split('/')[-1].split('_')[0]
            F2_name = csv_photo_pairs[csv_file].split('/')[-1].split('_')[2]
        else:
            photometry_full_path = None
            names = get_names(df_csv)
            F1_name = names[0]
            F2_name = names[1]

        df = df._append({'Date': parse_date(csv_file).date(), 'F1 Name': F1_name, 'F2 Name': F2_name, 'CSV': csv_full_path, 'Video': video_full_path, 'Photometry': photometry_full_path}, ignore_index=True)
    # Save df to CSV file
    if output_path is not None:
        output = os.path.join(output_path, 'matched_files.csv')
        df.to_csv(output, index=False)
    return df

## Class to load and access photometry and video data from a csv file, unifying data access for both photometry and video data
class PhotometryVideoData:
    ## Constructor for the PhotometryVideoData class
    # @param: photo_csv_path - string containing the path to the CSV file containing the photometry data
    # @param: df - pandas dataframe containing the CSV file, if not provided then the CSV file will be loaded from the photo_csv_path
    def __init__(self, photo_csv_path=None, df=None, trim_start=0) -> None:
        if photo_csv_path is not None and df is not None:
            raise ValueError("Error: Cannot provide both a CSV path and a dataframe")
        if photo_csv_path is None and df is None:
            raise ValueError("Error: Must provide either a CSV path or a dataframe")
        self.df = pd.read_csv(photo_csv_path) if df is None else df
        self.video_path = os.path.join(config.remote_path, self.df['Video Path'][0])
        self.csv_path = os.path.join(config.remote_path, self.df['CSV Path'][0])
        self.video = cv2.VideoCapture(self.video_path)
        self.csv = pd.read_csv(self.csv_path)
        self.sample_rate = self.df['Sample Rate'][0]
        self.video_fps = self.df['Video FPS'][0]
        self.F1_name = self.df['F1 Name'][0]
        self.F2_name = self.df['F2 Name'][0]
        self.trim_start = trim_start

    ## Get the photometry data for a given trial, getting all rows in the CSV file with the given trial number
    # @param: trial - int containing the trial number
    # @return: data - dict containing the photometry data for each channel
    def get_photometry_data(self, trial):
        data = dict()
        for column in self.df.columns:
            if re.match('405A|405C|465A|465C|Timestamp', column):
                data[column] = self.df[self.df['Trial'] == trial][column].to_numpy()[self.trim_start:]
        return data
    
    ## Get the time series for a given trial, calculated from the sample rate and the length of the trial
    # @param: trial - int containing the trial number
    # @return: time - numpy array containing the time series for the trial
    def get_time(self, trial):
        return np.linspace(0, len(self.get_photometry_data(trial)['405A']) / self.sample_rate, len(self.get_photometry_data(trial)['405A']))
    
    ## Get all trials in a given CSV file as dict of arrays, where each array is all trials stacked
    # @return: data - dict containing the photometry data for each channel
    def get_all_photometry_data(self):
        for trial in self.df['Trial'].unique():
            trial_data = self.get_photometry_data(trial)
            if trial == 0:
                data = dict.fromkeys(trial_data.keys(), None)
            for key in trial_data.keys():
                if data[key] is None:
                    data[key] = trial_data[key]
                else:
                    data[key] = np.vstack((data[key], trial_data[key]))
        return data

    ## Set the start trim to exclude transients
    # @param: trim_start - int containing the number of seconds to trim from the start
    def set_trim_start(self, trim_start):
        self.trim_start = trim_start
    
    ## Length of the experiment in # of trials
    # @return: length - int containing the number of trials in the experiment
    def __len__(self):
        return len(self.df['Trial'].unique())

    ## Get the video data for a given trial, finding the range of frames in the video corresponding to the trial
    ## then loading the frames from the video
    # @param: trial - int containing the trial number
    # @return: data - numpy array containing the video frames for the trial
    def get_video_data(self, trial):
        start_frame = self.df[self.df['Trial'] == trial]['Video Frame'].min()
        end_frame = self.df[self.df['Trial'] == trial]['Video Frame'].max()
        data = []
        self.video = cv2.VideoCapture(self.video_path)
        for i in range(0, int(start_frame)):
            self.video.read()
        for i in range(int(start_frame), int(end_frame)):
            ret, frame = self.video.read()
            data.append(frame)
        return np.array(data)[int(self.trim_start*self.video_fps / self.sample_rate):]
    
    ## get the video writer object for the video
    # @return: video - cv2 VideoWriter object
    def get_video_writer(self):
        return self.video
    
    ## Load video data from a range of time around a specified timestamp
    # @param: timestamp - float containing the timestamp to load the video data around
    # @param: window - float containing the window of time to load the video data around the timestamp in seconds
    # @return: data - numpy array containing the video frames for the specified time range
    def load_video_data_around_timestamp(self, timestamp, window):
        video_offset = self.df['Video Frame'][0]
        start_time = timestamp - window/2
        end_time = timestamp + window/2
        start_frame = start_time * self.video_fps + video_offset
        end_frame = end_time * self.video_fps + video_offset
        data = []
        for i in range(int(start_frame), int(end_frame)):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video.read()
            data.append(frame)
        return np.array(data)
    
    def plot_time_series_and_video(self, output_path, data, trial, sample_rate):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        video = self.get_video_data(trial)
        ax1.imshow(video[0])
        for dataset in data:
            timeseries = np.linspace(0, len(dataset) / sample_rate, len(dataset))
            ax2.plot(timeseries, dataset)

        line_max_bound = max([max(dataset) for dataset in data])
        line_min_bound = min([min(dataset) for dataset in data])

        def update(frame):
            ax1.clear()
            ax1.imshow(video[frame])
            ax2.clear()
            for dataset in data:
                timeseries = np.linspace(0, len(dataset) / sample_rate, len(dataset))
                ax2.plot(timeseries, dataset)
            ax2.vlines(((frame * sample_rate) // 30), line_min_bound, line_max_bound, color='r')
        
        ani = FuncAnimation(fig, update, interval=1000/30, frames=len(video))
        ani.save(output_path)
        

    
    ## Given a set of data and a video of different rates create a animation displaying the video and the data being plotted using matplotlib on the same figure
    def animate_trace(self, output_path, data, sample_rate, video, fps=30):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(video[0])
        ax2.plot(data)
        ## animate a vertical line and the video
        def animate(i):
            ax1.clear()
            ax1.imshow(video[i])
            ax2.vline(i*sample_rate//fps)
            ax2.set_xlim(0, len(data))
        
        ani = FuncAnimation(fig, animate, interval=1000/fps, frames=len(video))
        ani.save(output_path)

    
    ## Returns a row of the experiment CSV for a given trial
    # @param: trial - int containing the trial number
    # @return: data - pandas series containing the row of the CSV file
    def get_trial_data(self, trial):
        return self.csv.iloc[trial]




