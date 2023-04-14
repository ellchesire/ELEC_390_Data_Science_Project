import numpy as np
import h5py
import pandas as pd
import os

# define the directories and file names for the data
base_dir = os.getcwd()
walking_dir = os.path.join(base_dir, 'Walking_Data')
jumping_dir = os.path.join(base_dir, 'Jumping_Data')

matt_files = {
    'right_hand_walking': 'MrightHandW.csv',
    'left_back_pocket_walking': 'MleftBackPocketW.csv',
    'left_jacket_walking': 'MleftJacketW.csv',
    'right_front_pocket_walking': 'MrightFrontPocketW.csv',
    'right_jacket_walking': 'MrightJacketW.csv',
    'right_hand_jumping': 'MrightHandJ.csv',
    'left_back_pocket_jumping': 'MleftBackPocketJ.csv',
    'left_jacket_jumping': 'MleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'MrightFrontPocketJ.csv',
    'right_jacket_jumping': 'MrightJacketPocketJ.csv',
}

warren_files = {
    'right_hand_walking': 'WrightHandW.csv',
    'left_back_pocket_walking': 'WleftBackPocketW.csv',
    'left_jacket_walking': 'WleftJacketPocketW.csv',
    'right_front_pocket_walking': 'WrightFrontPocketW.csv',
    'right_jacket_walking': 'WrightJacketPocketW.csv',
    'right_hand_jumping': 'WrightHandJ.csv',
    'left_back_pocket_jumping': 'WleftBackPocketJ.csv',
    'left_jacket_jumping': 'WleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'WrightFrontPocketJ.csv',
    'right_jacket_jumping': 'WrightJacketPocketJ.csv',
}

ellen_files = {
    'right_hand_walking': 'ErightHandW.csv',
    'left_back_pocket_walking': 'EleftBackPocketW.csv',
    'left_jacket_walking': 'EleftJacketPocketW.csv',
    'right_front_pocket_walking': 'ErightFrontPocketW.csv',
    'right_jacket_walking': 'ErightJacketPocketW.csv',
    'right_hand_jumping': 'ErightHandJ.csv',
    'left_back_pocket_jumping': 'EleftBackPocketJ.csv',
    'left_jacket_jumping': 'EleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'ErightFrontPocketJ.csv',
    'right_jacket_jumping': 'ErightJacketPocketJ.csv',
}

all_files ={
    'right_hand_walking': 'MrightHandW.csv',
    'left_back_pocket_walking': 'MleftBackPocketW.csv',
    'left_jacket_walking': 'MleftJacketW.csv',
    'right_front_pocket_walking': 'MrightFrontPocketW.csv',
    'right_jacket_walking': 'MrightJacketW.csv',
    'right_hand_jumping': 'MrightHandJ.csv',
    'left_back_pocket_jumping': 'MleftBackPocketJ.csv',
    'left_jacket_jumping': 'MleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'MrightFrontPocketJ.csv',
    'right_jacket_jumping': 'MrightJacketPocketJ.csv',

}

# read in the data files for each subject and store them in a dictionary
matt_data = {}
for key, file in matt_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    matt_data[key] = df

warren_data = {}
for key, file in warren_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    warren_data[key] = df

ellen_data = {}
for key, file in ellen_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    ellen_data[key] = df

# combine the data into a single dataframe
matt_df = pd.concat(matt_data.values())
warren_df = pd.concat(warren_data.values())
ellen_df = pd.concat(ellen_data.values())
combined_df = pd.concat([matt_df, warren_df, ellen_df])
# print(combined_df)

combined_df.to_csv('test2.csv')




# store the combined data in an HDF5 file
with h5py.File('data.h5', 'w') as hdf:
    # Matt's Data Group
    Matt_Group = hdf.create_group('/Matt_Group')
    Matt_Group.create_dataset('matts', data=matt_df)

    # Warren's Data Group
    Warren_Group = hdf.create_group('/Warren_Group')
    tee = Warren_Group.create_dataset('Warren_data', data=warren_df)

    # Ellen's Data Group
    Ellen_Group = hdf.create_group('/Ellen_Group')
    Ellen_Group.create_dataset('Ellen_data', data=ellen_df)

    # Combining Datasets
    group = hdf.create_group('combined_dataset')
    # convert the pandas dataframe to a numpy array
    combined_data = combined_df.to_numpy()
    # write the numpy array to the HDF5 file
    comb = group.create_dataset('combined_data', data=combined_data)

    combined_df['Time (s)'] = pd.to_datetime((combined_df['Time (s)']))
    # print(combined_df['Time (s)'])


    # print(comb)