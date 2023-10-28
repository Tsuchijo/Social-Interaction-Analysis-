# Config folder to store references to all data files
# Path: config.py

# directory absolute path
import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
# Local paths
successful_trial_path = PROJECT_ROOT + '/data/successful_trials_by_day/'
media_path = PROJECT_ROOT + '/data/media/'
marker_path= PROJECT_ROOT + '/data/marker_reference/'
movement_labels_path = PROJECT_ROOT + '/data/movement_labels/'

# Remote paths
teamshare = '/mnt/teams/TM_Lab/Arjun Bhaskaran/'
remote_trial_path = teamshare + 'Social interaction project/Littermate interaction/csv files/'
remote_video_path = teamshare + 'Social interaction project/Littermate interaction/videos/'
remote_photometry_path = teamshare + 'Social interaction project/Littermate interaction/fiber photometry data/Mice with ID/'

# Output paths
video_output_path = '/mnt/teams/Tsuchitori/social_interaction_trials/'
movement_extracted_output_path = '/mnt/teams/Tsuchitori/social_interaction_trials_movement_extracted/'
photometry_output_path = '/mnt/teams/Tsuchitori/social_interaction_trials_photometry/'