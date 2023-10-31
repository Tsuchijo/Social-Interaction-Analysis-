# Overview
Repo for processing data from the Social Interaction protocol experiments

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [File Formatting](#file-formatting)
    - [Experiment CSV](#experiment-csv)

    - [Behavior Video](#behavior-video)
    - [Photometry Data](#photometry-data)



## Setup
After downloading follow these steps to make sure the repo is set up correctly

Pre-requisite software:

Miniconda: dl link - https://docs.conda.io/projects/miniconda/en/latest/


1. Change Paths in config.py to match your systems setup, the local paths should be the same but teamshare paths will be different depending on how your system is setup
    - 'teamshare' is the root of the teamshare path, change it to how teamshare is mounted on your system

2. setup required python environment 
    - First create a new conda environment, make sure conda is installed then, run ```conda create -n social_interaction python=3.10.12```
    - activate env with ```conda activate social_interaction``` then run ```conda install pip```
    - run ```pip install -r requirements.txt``` while in the project directory to install all required pacakges

## File Formatting

There are three main sources of data that the script accesses that need to have regularized naming conventions and structure those are:

- Behavior Video
- Experiment Data CSV
- Photometry Data

The Behavior video and CSV are both recorded onto the Rasberry PI, while the Photometry data is gotten from the TDT system, the expected format is as follows

### Experiment CSV

#### Title

The name has to contain the time that the recording started as a single long integer followed by ‘_log.csv’ in the format: 

```jsx
%year%month%day%hour%minute%second%(optional decimal time)
```

For example a trial recorded on 9-22-2023 at 9:57:39 would have the name 

```jsx
20230922095739_log.csv
```

#### Data

The CSV should contain columns with the following names and values 

- **cue_time in s :** the time the recoding time period for that row of data started in the same time format as the title, year month day hour minute second milis as a single long integer
- **[Mouse One / Two Name]_beambreak_time in s:** A timestamp containing the time one of the mice broke the beam, in the same time format as above, there should be one column for each mouse in the experiment, if the mouse did not break the beam then it should be left blank
- **[Mouse One / Two Name]_reward time in s:** A timestamp containing the time one of the mice was given an award, same time format as above

#### Example:

[20230922095739_log.csv](https://prod-files-secure.s3.us-west-2.amazonaws.com/abe77c5e-49ab-48c6-a705-4720574598f8/dd0966d7-2c4d-4ece-a828-c9fe9666af2e/20230922095739_log.csv)

### Video

#### Title

Each video of mice behavior should be labelled with the timestamp it was recorded in the same format as above delimited by underscores and in a h264 formate. For example:

```jsx
20230704151639_SI.h264
```

And should contain a full uncut video of the trial showing the area where the mice interact

### Fiber Photometry Data:

#### Title

There should only be 1 fiber photometry video per mouse pair per day, the directory containing the days recordings should have the format 

```jsx
Social_Interaction-[year][month][day]
```

where the date format is in the same integer format as used above. Additionally within each directory there should contain one folder for each pair of mice recorded that day with the format 

```jsx
[Mouse One Name]_F1_[Mouse Two Name]_F2
```

contained within which is the data output from the TDT fiber photometry system