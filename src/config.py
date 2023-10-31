# Config folder to store references to all data files
# Path: config.py

# directory absolute path
import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

# Remote paths
teamshare = '/mnt/teams/TM_Lab/Arjun Bhaskaran/Social interaction project/Littermate interaction/'
remote_trial_path = teamshare + 'csv files/'
remote_video_path = teamshare + 'videos/'
remote_photometry_path = teamshare + 'Littermate interaction/fiber photometry data/Mice with ID/'

# Output paths
teamshare + 'processed data/'