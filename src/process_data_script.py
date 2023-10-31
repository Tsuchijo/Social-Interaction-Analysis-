#!/home/murph_4090ws/Documents/Arjun_data/.conda/bin/python
import sys
import os
sys.path.append(os.path.abspath('../'))
import config
import Trial_Processing as pts

if __name__ == '__main__':
    df =  pts.match_datafiles_by_date(config.remote_trial_path, config.remote_video_path, config.remote_photometry_path, output_path=config.remote_processed_data_path)
    ## Iterate through all the rows in the dataframe and write the photometry data to a CSV file
    for row in df.iterrows():
        photo_path = row[1]['Photometry']
        csv_path = row[1]['CSV']
        video_path = row[1]['Video']
        names = row[1]['F1 Name'] +  row[1]['F2 Name']
        date = row[1]['Date'].strftime('%Y%m%d')
        output_path = date + '_' + names + '.csv'
        if photo_path is not None and csv_path is not None and video_path is not None:
            pts.write_photometry_to_csv(csv_path, video_path, photo_path, output_path, row[1]['F1 Name'], row[1]['F2 Name'], sample_rate=30, cue_offset_seconds=0)