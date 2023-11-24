# Config folder to store references to all data files
# Path: config.py

# directory absolute path
import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

## Remote paths
remote_path = '/mnt/teams/'
# Project root
teamshare = remote_path + 'TM_Lab/Arjun Bhaskaran/Social interaction project/October Test Data/'
remote_trial_path = teamshare + 'csv files/'
remote_video_path = teamshare + 'videos/'
remote_photometry_path = teamshare + 'fiber photometry data/'

# Output paths
remote_processed_data_path = teamshare + 'processed data/'