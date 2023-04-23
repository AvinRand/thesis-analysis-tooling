import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import pandas as pd
import seaborn as sns
import datetime
import numpy as np
import fnmatch
import glob
import cv2

# A module to streamline post-processing of retrospective analysis output data, mainly used for Organ-on-Chip Biosensing experiments, functions that extract optical data, consolidate data from fluigent log files, tabulate net shifts, etc. in Python are included.

# Avineet Randhawa | 2023 


def categorize_assay_data(layout: pd.DataFrame, data: pd.DataFrame, conditions_mapping: dict, readout: str, experimental_id: str, output_path: str) -> pd.DataFrame:
    '''This function organizes a 2-D array of data (e.g. a 96-well plate assay) based on an identically-sized array of identifying strings (e.g. a plate layout) using a user-populated dictionary of mappings.  The entries in the layout array should all consist of the same number of hyphen-delimited strings that correspond to a varied condition and value; it is critical that the order of strings in the layout array is consistent across entries (for example, if two conditions are varied in the experiment, such as antibody species and antibody concentration, each layout entry specifies each condition in the same order.  The dictionary keys each correspond to these hypnen-delimited strings, with the values comprising a tuple; the first item in the tuple is a string of the condition being varied (e.g. 'Antibody Concentration ng/mL'), while the second is the condition value (e.g. 10).  The order in which the dictionary is populated must reflect the order in which conditions are specified in layout strings.  The readout and experimental_id arguments refer to the type of data in the in array (e.g. 'Absorbance (p.d.u.)' and any string that identifies this experiment for later comparison, respectively.  The function returns a dataframe with headings corresponding to the varied conditions, the readout, and experimental identifier, and saves it to the directory specified by the output_path argument.))'''

    # Remove any unnamed columns if they exist (this is largely in case the first column of a microplate with alphabetical labels is included)
    for dataframe in [layout, data]:
        if "Unnamed" in dataframe[dataframe.columns[0]].name:
            dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
        else: 
            pass

    # Extract an ordered list of the unique conditions specified in the dictionary, which will constitute the headings of the output dataframe
    conditions_list = [value[0] for value in conditions_mapping.values() if value[0]]
    conditions_list = list(dict.fromkeys(conditions_list))

    organized_data = pd.DataFrame(columns=conditions_list + [readout] + ['Experimental Identifier'])

    # Loop through the columns and rows of the layout dataframe; for each entry, look up the hyphen-delimited string identifiers in the conditions mapping dictionary and assign the resulting condition values to a new list where the order of items reflects the position in the layout string (which should match the order in which conditions have been listed and defined in the dictionary). 
    for i, column in enumerate(layout):
        for j, row_entry in enumerate(layout[column]):
            label = layout[layout.columns[i]][j]
            data_point = data[data.columns[i]][j]
            organized_row = []
            for identifier in label.split('-'):
                if identifier in conditions_mapping.keys():
                    organized_row.append(conditions_mapping[identifier][1])
                else:
                    print(f'Identifying string in layout entry {label} (Column {column}, row {j+1}) not found in dictionary mapping')
                    break
            organized_data.loc[i * len(layout['1']) + j] = organized_row + [data_point] + [experimental_id]  
    
    organized_data.to_csv(output_path + "/Organized_Data.csv", index=False)

    return organized_data


def extract_peak_positions(path_to_sweep_output_folder: str) -> pd.DataFrame:
    '''This function extracts peak position profiles when the updated retrospective analysis script (with a GUI) is used for analysis.  It assumes that results are provided in nanometers, the power arrays of each sensor contains the word power (case sensitive) and are arranged in ascending order (i.e. power_1_array, power_2_array, etc.), and that the timesteps array units are in minutes; the output is placed into a dataframe'''

    sensor_matrix = scipy.io.loadmat(path_to_sweep_output_folder + '/peak_shifts.mat')
    sensor_data = pd.DataFrame()

    i = 1

    for key in sensor_matrix:
        if "power" in key:
            sensor_data.insert(loc = i-1, column = f'Sensor {i}', value = sensor_matrix[f'{key}'][0])
            i += 1
        if "t_" in key:
            sensor_data.insert(loc = i-1, column = f'Time', value = sensor_matrix[f'{key}'][0])

    for column in sensor_data:
        if 'Sensor' in column:
            sensor_data[column] -= sensor_data[column].iloc[0]

    return sensor_data


def consolidate_fluigent_data(sensor_dataframe: pd.DataFrame, path_to_directory: str, save_path: str, experimental_id: str, protocol_keyword: str = None, multi_part: bool = False, delimiter: str = ',') -> pd.DataFrame:
    '''This function takes the power sensor dataframe and adds columns corresponding to data from the fluigent log file.  The fluigent logging frequency will likely not match the sweep frequency, and so this function assigns the fluigent log datapoint immediately before a given sweep to that entry in the resulting dataframe, thereby providing a consolidated dataframe with corresponding optical and fluidic information with an added column that identifies the experiment based on a user-specified string (defined by the experimental_id argument). The path to directory argument should define the directory in which the fluigent log CSV file resides; if there are multiple CSVs, define the optional protocol_keyword argument as a string unique to the filename of that CSV. If your assay requires multiple stages, include 'Part 1' and 'Part 2' in the CSV and set the optional multi_part parameter as True. '''

    if multi_part:
        # Load the log files of interest based on keywords in the filename (based on the name assigned to the fluigent protocol running)
        fluigent_part_1 = pd.read_csv(glob.glob(path_to_directory + "/*Part-1*.csv")[0], delimiter=delimiter)
        fluigent_part_2 = pd.read_csv(glob.glob(path_to_directory + "/*Part-2*.csv")[0], delimiter=delimiter)
        # Convert the fluigent timestamps to timestamps recognized by the pandas library for downstream processing
        fluigent_part_1['Time'] = pd.to_datetime(fluigent_part_1['Time'])
        fluigent_part_2['Time'] = pd.to_datetime(fluigent_part_2['Time'])
        # Convert fluigent timestamps from absolute to relative time for compatibility with sweep timesteps
        start_datetime = fluigent_part_1['Time'][0]
        fluigent_part_1['Time'] = fluigent_part_1['Time'] - start_datetime
        fluigent_part_2['Time'] = fluigent_part_2['Time'] - start_datetime
        # Convert relative timestamps to minutes
        fluigent_part_1['Time'] = fluigent_part_1['Time'].dt.total_seconds()
        fluigent_part_1['Time'] = fluigent_part_1['Time']/60
        fluigent_part_2['Time'] = fluigent_part_2['Time'].dt.total_seconds()
        fluigent_part_2['Time'] = fluigent_part_2['Time']/60
        # Consolidate timesteps across two consecutive fluigent protocols into new dataframe
        fluigent = pd.concat([fluigent_part_1, fluigent_part_2], ignore_index=True, sort=False)

    else:
        fluigent = pd.read_csv(glob.glob(path_to_directory + f"/*{protocol_keyword}*.csv")[0], delimiter=delimiter)
        fluigent['Time'] = pd.to_datetime(fluigent['Time'])
        start_datetime = fluigent['Time'][0]
        fluigent['Time'] = fluigent['Time'] - start_datetime
        fluigent['Time'] = fluigent['Time'].dt.total_seconds()
        fluigent['Time'] = fluigent['Time']/60

    # Loop through sweep timesteps and extract the fluigent entry nearest in time (but prior) to each sweep timestep
    union_indices = [0]
    for sensor_time in sensor_dataframe['Time'][1:]:
        lower_index = fluigent[fluigent['Time'] < sensor_time]['Time'].idxmax()
        union_indices.append(lower_index)

    # Filter consolidated fluigent log file for those entries identified just above
    fluigent_data = fluigent.loc[union_indices]

    # Consolidate the relevent columns of the filtered fluigent log file (renaming said column names more intuitively) with peak positions dataframe
    all_data = sensor_dataframe.copy()
    all_data['Valve Position (Channel 1)'] = fluigent_data['M-Switch #1 [Switch EZ #3 (10535)]'].to_numpy()
    all_data['Valve Position (Channel 2)'] = fluigent_data['M-Switch #4 [Switch EZ #3 (10535)]'].to_numpy()
    all_data['Flowrate (Channel 1)'] = fluigent_data['Flow EZ #1 (13109) Q'].to_numpy()
    all_data['Flowrate (Channel 2)'] = fluigent_data['Flow EZ #2 (13110) Q'].to_numpy()
    all_data['Pressure (Channel 1)'] = fluigent_data['Flow EZ #1 (13109)'].to_numpy()
    all_data['Pressure (Channel 2)'] = fluigent_data['Flow EZ #2 (13110)'].to_numpy()
    all_data['Experimental Identifier'] = experimental_id

    all_data.to_csv(save_path+'/Tabulated Data.csv', index = False)

    return all_data


def tabulate_net_peak_shifts(experiment_data: pd.DataFrame, conditions_mapping: dict, sweep_coordinates: list, list_of_assay_stages: list, experimental_id: str, save_path: str, sweeps_average: int = 5) -> pd.DataFrame:
    '''This function takes six mandatory and one optional argument to return a tabulated list of calculated net resonance peak shifts as a pandas dataframe; the arguments are, respectively, the experiment data (generally the output of the 'consolidate_fluigent_data' function or a dataframe with column names in the format 'Sensor 1', 'Sensor 2' ... and 'Time'), the user-specified conditions mapping (a nested dictionary mapping each sensor to a dictionary of parameter-value pairs), sweep coordinates (sweep number integers from which to calculate net shift differences), the stages of the assay (strings, whose should match the number of coordinates), an experimental identifier (e.g. the experiment title), and a save path (into which the output dataframe will be stored).  The optional argument defines the number of sweeps across which an average will be taken for net peak shift calculations (the default is 5, meaning the peak position will be the average of a coordinate and the five sweeps following it).'''
    
    quantified_peak_pos_data = pd.DataFrame(columns= ['Optical Channel'] + list(list(conditions_mapping.values())[0].keys()) + ['Assay Stage', 'Mean Peak Position Over 5 Sweeps (nm)'])
    
    if len(sweep_coordinates) == len(list_of_assay_stages):
        for i, sensor in enumerate(conditions_mapping.keys()):
            optical_channel = [int(s) for s in sensor.split() if s.isdigit()]
            for j, sweep in enumerate(sweep_coordinates):
                mean_peak_pos = experiment_data[sensor].iloc[sweep:sweep+sweeps_average].mean()
                stage = list_of_assay_stages[j]
                count = i * len(sweep_coordinates) + j
                quantified_peak_pos_data.loc[count] = optical_channel + list(conditions_mapping[sensor].values()) + [stage, mean_peak_pos]
    else:
        print('The number of coordinates provided does not match the number of assay stages')
    
    if len(quantified_peak_pos_data) > 0:
        drop_indexes = quantified_peak_pos_data.index[quantified_peak_pos_data['Assay Stage'] == list_of_assay_stages[0]].tolist()
        shift_data = quantified_peak_pos_data.copy()
        shift_data["Peak Shift"] = shift_data['Mean Peak Position Over 5 Sweeps (nm)'].diff()
        shift_data.drop(drop_indexes, inplace=True)
        shift_data.drop(['Mean Peak Position Over 5 Sweeps (nm)'], axis = 1, inplace=True)
        shift_data["Experimental Identifier"] = experimental_id
        shift_data.to_csv(save_path + '/Processed Shift Data.csv', index=False)
        return shift_data


def stich_video_from_timelapse_captures(image_format, video_framerate, image_folder, video_name, save_path):
    '''This funciton uses openCV's VideoWriter to stich a video from a series of images (intended to be timelapse captures from the overhead microscope for observation of potential bubbles).  The image format variable should be in the following format: '.format', (e.g. '.jpg').  The video name should include the format (mp4), i.e. 'video_name.mp4'.'''

    images = [img for img in os.listdir(image_folder) if img.endswith(image_format)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path + '/' + video_name, fourcc, video_framerate, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()