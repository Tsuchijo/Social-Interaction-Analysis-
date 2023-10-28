#!/home/murph_4090ws/Documents/Arjun_data/.conda/bin/python

# Import libraries
import os 
import sys
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import math
import argparse
import pickle
import cv2
import tdt
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
# @return: 405A, 405B, 465A, 465B - N x L numpy arrays containing the data from each channel, where N is the number of trials and L is the length of the trial
def load_photometry_streams(path, sample_rate=None, trial_duration=30, skip_trials=1):
    # Load the TDT data from the path (this may take a while)
    data = tdt.read_block(path)

    if sample_rate is None:
        sample_rate = data.streams['_405A'].fs

    # Get the data from each channel, downsample it to samplerate if needed
    stream_405A = downsample(data.streams['_405A'].data, data.streams['_405A'].fs, sample_rate)
    stream_405C = downsample(data.streams['_405C'].data, data.streams['_405C'].fs, sample_rate)
    stream_465A = downsample(data.streams['_465A'].data, data.streams['_465A'].fs, sample_rate)
    stream_465C = downsample(data.streams['_465C'].data, data.streams['_465C'].fs, sample_rate)

    # All streams must be the same length
    assert len(stream_405A) == len(stream_405C) == len(stream_465A) == len(stream_465C), "Error: Photometry streams are not the same length"

    # Sum the data together from each channel and then detect the rising edges of the data
    total_stream_405 = stream_405A + stream_405C + stream_465A + stream_465C
    _ , start_indices, end_indices = detect_triggers(total_stream_405, threshold=20, window=int(sample_rate))

    print("Number of trials: " + str(len(start_indices)))

    # Split the data into trials based on the rising edges and the trial duration
    # Skips the Last trial because it is usually cut off
    try:
        trials_405A = np.array([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_405A, start_indices)[skip_trials:-1]])
        trials_405C = np.array([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_405C, start_indices)[skip_trials:-1]])
        trials_465A = np.array([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_465A, start_indices)[skip_trials:-1]])
        trials_465C = np.array([ trial[:int(trial_duration*sample_rate)] for trial in np.split(stream_465C, start_indices)[skip_trials:-1]])
        return trials_405A, trials_405C, trials_465A, trials_465C

    except IndexError:
        print("Error: Photometry data at path: " + path + " is shorter than the given trial duration, either trial duration is too long or the threshold is incorrect")
        return None, None, None, None
    

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


if __name__ == '__main__':
    df =  match_datafiles_by_date(config.remote_trial_path, config.remote_video_path, config.remote_photometry_path, output_path='.')
    ## Get the first photometry path for testing 
    photo_path = df['Photometry'][1]
    csv_path = df['CSV'][1]
    ## Load the photometry data
    trials_405A, trials_405C, trials_465A, trials_465C = load_photometry_streams(photo_path, sample_rate=24)
    ## Plot the photometry data
    print(trials_405A.shape)
    plt.plot(trials_465A[:, :].T)
    plt.show()