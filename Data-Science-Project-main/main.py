import os
import io
import random
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import pandas as pd
from scipy.stats import skew, kurtosis
import statistics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tkinter as tk
from tkinter.filedialog import askopenfilename
from sklearn.linear_model import LogisticRegression
from tkinter import *
from sklearn.model_selection import train_test_split, learning_curve

np.random.seed(90)

#changing all csv files into DataFrames

# Matt's Data
MrightHandW = pd.read_csv('Walking_Data/MrightHandW.csv')
MleftBackPocketW = pd.read_csv('Walking_Data/MleftBackPocketW.csv')
MleftJacketPocketW = pd.read_csv('Walking_Data/MleftJacketW.csv')
MrightFrontPocketW = pd.read_csv('Walking_Data/MrightFrontPocketW.csv')
MrightJacketPocketW = pd.read_csv('Walking_Data/MrightJacketW.csv')

MrightHandJ = pd.read_csv('Jumping_Data/MrightHandJ.csv')
MleftBackPocketJ = pd.read_csv('Jumping_Data/MleftBackPocketJ.csv')
MleftJacketPocketJ = pd.read_csv('Jumping_Data/MleftJacketPocketJ.csv')
MrightFrontPocketJ = pd.read_csv('Jumping_Data/MrightFrontPocketJ.csv')
MrightJacketPocketJ = pd.read_csv('Jumping_Data/MrightJacketPocketJ.csv')

# Warren's Data
WrightHandW = pd.read_csv('Walking_Data/WrightHandW.csv')
WleftBackPocketW = pd.read_csv('Walking_Data/WleftBackPocketW.csv')
WleftJacketPocketW = pd.read_csv('Walking_Data/WleftJacketPocketW.csv')
WrightFrontPocketW = pd.read_csv('Walking_Data/WrightFrontPocketW.csv')
WrightJacketPocketW = pd.read_csv('Walking_Data/WrightJacketPocketW.csv')

WrightHandJ = pd.read_csv('Jumping_Data/WrightHandJ.csv')
WleftBackPocketJ = pd.read_csv('Jumping_Data/WleftBackPocketJ.csv')
WleftJacketPocketJ = pd.read_csv('Jumping_Data/WleftJacketPocketJ.csv')
WrightFrontPocketJ = pd.read_csv('Jumping_Data/WrightFrontPocketJ.csv')
WrightJacketPocketJ = pd.read_csv('Jumping_Data/WrightJacketPocketJ.csv')

# Ellen's Data
ErightHandW = pd.read_csv('Walking_Data/ErightHandW.csv')
EleftBackPocketW = pd.read_csv('Walking_Data/EleftBackPocketW.csv')
EleftJacketPocketW = pd.read_csv('Walking_Data/EleftJacketPocketW.csv')
ErightFrontPocketW = pd.read_csv('Walking_Data/ErightFrontPocketW.csv')
ErightJacketPocketW = pd.read_csv('Walking_Data/ErightJacketPocketW.csv')

ErightHandJ = pd.read_csv('Jumping_Data/ErightHandJ.csv')
EleftBackPocketJ = pd.read_csv('Jumping_Data/EleftBackPocketJ.csv')
EleftJacketPocketJ = pd.read_csv('Jumping_Data/EleftJacketPocketJ.csv')
ErightFrontPocketJ = pd.read_csv('Jumping_Data/ErightFrontPocketJ.csv')
ErightJacketPocketJ = pd.read_csv('Jumping_Data/ErightJacketPocketJ.csv')

#Making Datasets for every group member

Matts_Dataset = pd.concat([MleftBackPocketW, MleftJacketPocketW, MrightFrontPocketW, MrightHandW, MrightJacketPocketW,
                           MleftBackPocketJ, MleftJacketPocketJ, MrightFrontPocketJ, MrightHandJ, MrightJacketPocketJ])

Matts_Dataset.to_csv('Data/Matts_Data.csv')

Warrens_Dataset = pd.concat([WleftBackPocketW, WleftJacketPocketW, WrightFrontPocketW, WrightHandW, WrightJacketPocketW,
                             WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ,
                             WrightJacketPocketJ])

Warrens_Dataset.to_csv('Data/Warrens_Data.csv')

Ellens_Dataset = pd.concat([EleftBackPocketW, EleftJacketPocketW, ErightFrontPocketW, ErightHandW, ErightJacketPocketW,
                            EleftBackPocketJ, EleftJacketPocketJ, ErightFrontPocketJ, ErightHandJ, ErightJacketPocketJ])

Ellens_Dataset.to_csv('Data/Ellens_Data.csv')

#Making a Jumping and Walking Dataset
Walk_Dataset = pd.concat([MleftBackPocketW, MleftJacketPocketW, MrightFrontPocketW, MrightHandW, MrightJacketPocketW,
WleftBackPocketW, WleftJacketPocketW, WrightFrontPocketW, WrightHandW, WrightJacketPocketW,
EleftBackPocketW, EleftJacketPocketW, ErightFrontPocketW, ErightHandW, ErightJacketPocketW])
Walk_Dataset['label']=0


Jump_Dataset = pd.concat([MleftBackPocketJ, MleftJacketPocketJ, MrightFrontPocketJ, MrightHandJ, MrightJacketPocketJ,
WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ, WrightJacketPocketJ,
EleftBackPocketJ, EleftJacketPocketJ, ErightFrontPocketJ, ErightHandJ, ErightJacketPocketJ])
Jump_Dataset['label']=1

#Making a set containing all of the data
Full_set = pd.concat([Jump_Dataset, Walk_Dataset], ignore_index=True)
Full_set.to_csv('Data/Labeled_Full.csv', index=False)

combined_data = pd.concat([
    EleftBackPocketW,
    EleftJacketPocketW,
    ErightFrontPocketW,
    ErightHandW,
    ErightJacketPocketW,
    MleftBackPocketW,
    MleftJacketPocketW,
    MrightFrontPocketW,
    MrightHandW,
    MrightJacketPocketW,
    WleftBackPocketW,
    WleftJacketPocketW,
    WrightFrontPocketW,
    WrightHandW,
    WrightJacketPocketW,
    EleftBackPocketJ,
    EleftJacketPocketJ,
    ErightFrontPocketJ,
    ErightHandJ,
    ErightJacketPocketJ,
    MleftBackPocketJ,
    MleftJacketPocketJ,
    MrightFrontPocketJ,
    MrightHandJ,
    MrightJacketPocketJ,
    WleftBackPocketJ,
    WleftJacketPocketJ,
    WrightFrontPocketJ,
    WrightHandJ,
    WrightJacketPocketJ,
])

combined_data.to_csv('Data/Combined_Dataset.csv', index=False)

# Visualization, making all the graphs
startTime = 3000
endTime = 3500
smallStart = 3000
smallEnd = 3300

time = combined_data.iloc[startTime:endTime, 0]
timeSmall = combined_data.iloc[smallStart: smallEnd, 0]

#Commented out for the final product

# fig11, ax11 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Full Set')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Acceleration (m/s^2)', fontsize=15)
# ax11.scatter(Full_set['Time (s)'], Full_set['Absolute acceleration (m/s^2)'], label='Full Set', color='red')
# ax11.scatter(Full_set['Time (s)'], Full_set['label'], label='Full Set Label', color='blue')
# plt.legend(loc="upper left")
#
# # plotting left back pocket absolute accel
# fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Absolute Accleration Left Back Pocket Walking and Jumping (All)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Acceleration (m/s^2)', fontsize=15)
# ax.plot(time, EleftBackPocketW.iloc[startTime: endTime, 4], label='Ellen', color='red')
# ax.plot(time, WleftBackPocketW.iloc[startTime: endTime, 4], label='Warren', color='blue')
# ax.plot(time, MleftBackPocketW.iloc[startTime: endTime, 4], label='Matthew', color='green')
#
# ax.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 4], label='Ellen Jump', color='red', linestyle='dashed')
# ax.plot(time, WleftBackPocketJ.iloc[startTime: endTime, 4], label='Warren Jump', color='blue', linestyle='dashed')
# ax.plot(time, MleftBackPocketJ.iloc[startTime: endTime, 4], label='Matthew Jump', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting left back pocket xyz accel [Ellen]
# fig1, ax1 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('XYZ-Axis Left Back Pocket Walking and Jumping (Ellen)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 1], label='X', color='red')
# ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 2], label='Y', color='blue')
# ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 3], label='Z', color='green')
#
# ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
# ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
# ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
# # plotting hand xyz accel [Warren]
# fig2, ax2 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('XYZ-Axis Hand Walking and Jumping (Warren)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax2.plot(time, WrightHandW.iloc[startTime: endTime, 1], label='X', color='red')
# ax2.plot(time, WrightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
# ax2.plot(time, WrightHandW.iloc[startTime: endTime, 3], label='Z', color='green')
#
# ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
# ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
# ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting hand xyz accel [Ellen]
# fig3, ax3 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('XYZ-Axis Hand Walking and Jumping (Ellen)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax3.plot(time, ErightHandW.iloc[startTime: endTime, 1], label='X', color='red')
# ax3.plot(time, ErightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
# ax3.plot(time, ErightHandW.iloc[startTime: endTime, 3], label='Z', color='green')
#
# ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
# ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
# ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting hand xyz accel [Matthew]
# fig4, ax4 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('XYZ-Axis Hand Walking and Jumping (Matthew)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
# ax4.plot(time, MrightHandW.iloc[startTime: endTime, 1], label='X', color='red')
# ax4.plot(time, MrightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
# ax4.plot(time, MrightHandW.iloc[startTime: endTime, 3], label='Z', color='green')
#
# ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
# ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
# ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting x-axis of right front pocket of everyone
# fig5, ax5 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('X-Axis Right Front Walking and Jumping Pocket (All)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('X Acceleration (m/s^2)', fontsize=15)
# ax5.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 1], label='M', color='red')
# ax5.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 1], label='W', color='blue')
# ax5.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 1], label='E', color='green')
#
# ax5.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='M', color='red', linestyle='dashed')
# ax5.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='W', color='blue', linestyle='dashed')
# ax5.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='E', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting y-axis of right front pocket data of everyone
# fig6, ax6 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Y-Axis Right Front Walking and Jumping Pocket (All)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Y Acceleration (m/s^2)', fontsize=15)
# ax6.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 2], label='M', color='red')
# ax6.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 2], label='W', color='blue')
# ax6.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 2], label='E', color='green')
#
# ax6.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='M', color='red', linestyle='dashed')
# ax6.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='W', color='blue', linestyle='dashed')
# ax6.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='E', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#
#
# # plotting z-axis of right front pocket data
# fig7, ax7 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Z-Axis Right Front Walking and Jumping Pocket (All)')
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Z Acceleration (m/s^2)', fontsize=15)
# ax7.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 3], label='M', color='red')
# ax7.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 3], label='W', color='blue')
# ax7.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 3], label='E', color='green')
#
# ax7.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='M', color='red', linestyle='dashed')
# ax7.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='W', color='blue', linestyle='dashed')
# ax7.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='E', color='green', linestyle='dashed')
# plt.legend(loc="upper left")
#

#Meta Data Graphs
# metaTimeR = pd.read_csv('right_hand_walking_2023-03-09_16-56-21 (1)/meta/time.csv')
# metaTimeW = pd.read_csv('walking_back_left_pocket_2023-03-09_16-56-36 (1)/meta/time.csv')
#
#
# fig11, ax11 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Time Meta Data for Right Hand Walking [Ellen] vs Walking Back Left Pocket [Ellen]')
# plt.xlabel('Experiement Time', fontsize=15)
# plt.ylabel('Time', fontsize=15)
# ax11.plot( metaTimeR.iloc[:, 1], label='right hand walking', color='red')
# ax11.plot(metaTimeW.iloc [:, 1],  metaTimeW.iloc [:, 1], label='back pocket walking', color='green')
# plt.legend(loc="upper left")
#
# fig11, ax11 = plt.subplots(figsize=(10, 10), layout="constrained")
# plt.title('Time Meta Data for Right Hand Walking [Ellen] vs Walking Back Left Pocket [Ellen]')
# plt.xlabel('System Time', fontsize=15)
# plt.ylabel('Time', fontsize=15)
# ax11.plot( metaTimeR.iloc[:, 2], label='right hand walking', color='red')
# ax11.plot(metaTimeW.iloc [:, 2],  label='back pocket walking', color='green')
# plt.legend(loc="upper left")
#


#Function App calls to make classifer + processing input csv and giving predicted label
def analyzedata(text):

    #Making a DataFrame with the input csv

    dataIn = pd.read_csv(io.StringIO(text), sep=',')

    np.random.seed(90)

    #redoing all of the data classification for the app for the classifier
    # Matt's Data
    MrightHandW = pd.read_csv('Walking_Data/MrightHandW.csv')
    MleftBackPocketW = pd.read_csv('Walking_Data/MleftBackPocketW.csv')
    MleftJacketPocketW = pd.read_csv('Walking_Data/MleftJacketW.csv')
    MrightFrontPocketW = pd.read_csv('Walking_Data/MrightFrontPocketW.csv')
    MrightJacketPocketW = pd.read_csv('Walking_Data/MrightJacketW.csv')

    MrightHandJ = pd.read_csv('Jumping_Data/MrightHandJ.csv')
    MleftBackPocketJ = pd.read_csv('Jumping_Data/MleftBackPocketJ.csv')
    MleftJacketPocketJ = pd.read_csv('Jumping_Data/MleftJacketPocketJ.csv')
    MrightFrontPocketJ = pd.read_csv('Jumping_Data/MrightFrontPocketJ.csv')
    MrightJacketPocketJ = pd.read_csv('Jumping_Data/MrightJacketPocketJ.csv')

    # Warren's Data
    WrightHandW = pd.read_csv('Walking_Data/WrightHandW.csv')
    WleftBackPocketW = pd.read_csv('Walking_Data/WleftBackPocketW.csv')
    WleftJacketPocketW = pd.read_csv('Walking_Data/WleftJacketPocketW.csv')
    WrightFrontPocketW = pd.read_csv('Walking_Data/WrightFrontPocketW.csv')
    WrightJacketPocketW = pd.read_csv('Walking_Data/WrightJacketPocketW.csv')

    WrightHandJ = pd.read_csv('Jumping_Data/WrightHandJ.csv')
    WleftBackPocketJ = pd.read_csv('Jumping_Data/WleftBackPocketJ.csv')
    WleftJacketPocketJ = pd.read_csv('Jumping_Data/WleftJacketPocketJ.csv')
    WrightFrontPocketJ = pd.read_csv('Jumping_Data/WrightFrontPocketJ.csv')
    WrightJacketPocketJ = pd.read_csv('Jumping_Data/WrightJacketPocketJ.csv')

    # Ellen's Data
    ErightHandW = pd.read_csv('Walking_Data/ErightHandW.csv')
    EleftBackPocketW = pd.read_csv('Walking_Data/EleftBackPocketW.csv')
    EleftJacketPocketW = pd.read_csv('Walking_Data/EleftJacketPocketW.csv')
    ErightFrontPocketW = pd.read_csv('Walking_Data/ErightFrontPocketW.csv')
    ErightJacketPocketW = pd.read_csv('Walking_Data/ErightJacketPocketW.csv')

    ErightHandJ = pd.read_csv('Jumping_Data/ErightHandJ.csv')
    EleftBackPocketJ = pd.read_csv('Jumping_Data/EleftBackPocketJ.csv')
    EleftJacketPocketJ = pd.read_csv('Jumping_Data/EleftJacketPocketJ.csv')
    ErightFrontPocketJ = pd.read_csv('Jumping_Data/ErightFrontPocketJ.csv')
    ErightJacketPocketJ = pd.read_csv('Jumping_Data/ErightJacketPocketJ.csv')

    Matts_Dataset = pd.concat(
        [MleftBackPocketW, MleftJacketPocketW, MrightFrontPocketW, MrightHandW, MrightJacketPocketW,
         MleftBackPocketJ, MleftJacketPocketJ, MrightFrontPocketJ, MrightHandJ, MrightJacketPocketJ])


    Warrens_Dataset = pd.concat(
        [WleftBackPocketW, WleftJacketPocketW, WrightFrontPocketW, WrightHandW, WrightJacketPocketW,
         WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ,
         WrightJacketPocketJ])


    Ellens_Dataset = pd.concat(
        [EleftBackPocketW, EleftJacketPocketW, ErightFrontPocketW, ErightHandW, ErightJacketPocketW,
         EleftBackPocketJ, EleftJacketPocketJ, ErightFrontPocketJ, ErightHandJ, ErightJacketPocketJ])


    Walk_Dataset = pd.concat(
        [MleftBackPocketW, MleftJacketPocketW, MrightFrontPocketW, MrightHandW, MrightJacketPocketW,
         WleftBackPocketW, WleftJacketPocketW, WrightFrontPocketW, WrightHandW, WrightJacketPocketW,
         EleftBackPocketW, EleftJacketPocketW, ErightFrontPocketW, ErightHandW, ErightJacketPocketW])
    Walk_Dataset['label'] = 0

    Jump_Dataset = pd.concat(
        [MleftBackPocketJ, MleftJacketPocketJ, MrightFrontPocketJ, MrightHandJ, MrightJacketPocketJ,
         WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ, WrightJacketPocketJ,
         EleftBackPocketJ, EleftJacketPocketJ, ErightFrontPocketJ, ErightHandJ, ErightJacketPocketJ])
    Jump_Dataset['label'] = 1

    Full_set = pd.concat([Jump_Dataset, Walk_Dataset], ignore_index=True)

    combined_data = pd.concat([
        EleftBackPocketW,
        EleftJacketPocketW,
        ErightFrontPocketW,
        ErightHandW,
        ErightJacketPocketW,
        MleftBackPocketW,
        MleftJacketPocketW,
        MrightFrontPocketW,
        MrightHandW,
        MrightJacketPocketW,
        WleftBackPocketW,
        WleftJacketPocketW,
        WrightFrontPocketW,
        WrightHandW,
        WrightJacketPocketW,
        EleftBackPocketJ,
        EleftJacketPocketJ,
        ErightFrontPocketJ,
        ErightHandJ,
        ErightJacketPocketJ,
        MleftBackPocketJ,
        MleftJacketPocketJ,
        MrightFrontPocketJ,
        MrightHandJ,
        MrightJacketPocketJ,
        WleftBackPocketJ,
        WleftJacketPocketJ,
        WrightFrontPocketJ,
        WrightHandJ,
        WrightJacketPocketJ,
    ])

    #Checking to see the CSV exists
    if os.path.exists("Data/Features_Combined.csv"):
        os.remove("Data/Features_Combined.csv")
        print("File deleted")
    if os.path.exists("Data/Input_Features.csv"):
        os.remove("Data/Input_Features.csv")
        print("Input File Deleted")

    with h5py.File('data.h5', 'w') as hdf:
        # Combined Dataset Creation
        combined_DataSet = hdf.create_group('/mainDataset')
        combined_DataSet.create_dataset('mainDataset', data=Full_set)


        # Cleaning (Pre Processing), window of 60 to find the rolling average
        for column in Full_set.columns:
            Full_set[column] = Full_set[column].rolling(window=60).mean()

        for column in dataIn.columns:
            #other
            dataIn[column] =  dataIn[column].rolling(window=60).mean()


        # Drop rows with missing values
        Full_set.dropna(inplace=True)
        dataIn.dropna(inplace=True)

        # Save preprocessed data
        Full_set.to_csv('Data/Combined_Dataset_preprocessed.csv', index=False)

        window_size = 500

        # Segmenting Data
        segments = [Full_set.iloc[i:i + window_size] for i in range(0, len(Full_set), window_size)]
        count_segments = int(np.ceil(len(Full_set) / window_size))

        # Segmenting Input data

        segmentsIn = [dataIn.iloc[i:i + window_size] for i in range(0, len(dataIn), window_size)]
        count_segmentsIn = int(np.ceil(len(dataIn) / window_size))
        print(count_segmentsIn)

        #Feature Extraction for Input Data
        for i in range(count_segmentsIn):

            dataframe = pd.DataFrame(segmentsIn[i])

            # Normalizing the data (Pre Processing)
            scaler = preprocessing.StandardScaler()
            dfIn = pd.DataFrame(data=scaler.fit_transform(dataframe.iloc[:,:]))


            feature_namesIn = ['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance', 'Skew', 'Kurtosis',
                             'Standard Deviation', 'Inter Quartile Range']

            featsIn = pd.DataFrame(columns=feature_namesIn)

            # Extracting Features from Absolute Acceleration
            featuresIn = [dfIn.max().values,
                        dfIn.min().values,
                        dfIn.max().values - dfIn.min().values,
                        dfIn.mean().values,
                        dfIn.median().values,
                        dfIn.var().values,
                        dfIn.skew().values,
                        dfIn.kurt().values,
                        dfIn.std(axis=1).values,
                        dfIn.quantile(0.75).values - dfIn.quantile(0.25).values]

            featsIn.loc[i] = [featuresIn[j][4] for j in range(len(featuresIn))]

            #Writing the features Dataframe into a CSV
            with open('Data/Input_Features.csv', 'a') as f:
                featsIn.to_csv(f, header=f.tell() == 0, index=False)

            segmentsIn[i] = segmentsIn[i].sample(frac=1).reset_index(drop=True)
        #Feature Extraction for the collected Data for the Model
        #Same as above for the steps
        for i in range(count_segments):

            dataframe = pd.DataFrame(segments[i])


            # Normalizing (Pre Processing)
            scaler = preprocessing.StandardScaler()
            df2 = pd.DataFrame(data=scaler.fit_transform(dataframe.iloc[:, :-1]))

            df2.to_csv("Data/PreProcessedD.csv", index=False)

            #Graph for Visualization
            # fig11, ax11 = plt.subplots(figsize=(10, 10), layout="constrained")
            # plt.title('Preprocessed Data Absolute Acceleration')
            # plt.xlabel('Time (s)', fontsize=15)
            # plt.ylabel('Acceleration', fontsize=15)
            # ax11.plot(df[0], df[4])
            #

            feature_names = ['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance', 'Skew', 'Kurtosis', 'Standard Deviation', 'Inter Quartile Range']

            feats = pd.DataFrame(columns=feature_names)

            # Feature Extraction Part 2
            features = [df2.max().values,
                            df2.min().values,
                            df2.max().values-df2.min().values,
                            df2.mean().values,
                            df2.median().values,
                            df2.var().values,
                            df2.skew().values,
                            df2.kurt().values,
                            df2.std(axis=1).values,
                            df2.quantile(0.75).values-df2.quantile(0.25).values]
            feats.loc[i] = [features[j][4] for j in range(len(features))]

            with open('Data/Features_Combined.csv', 'a') as f:
                feats.to_csv(f, header=f.tell() == 0, index=False)

            # Data Shuffling
            segments[i] = segments[i].sample(frac=1).reset_index(drop=True)

            # HDF organization for the Data which is commented out for testing as
            # it throws errors when the groups already exist

            # Training Dataset Creation
            # training_Dataset = hdf.create_group('/mainDataset/Training')
            # training_Dataset.create_dataset('training_dataset', data=train_data)

            # Testing Dataset Creation
            # testing_Dataset = hdf.create_group('/mainDataset/Testing')
            # testing_Dataset.create_dataset('testing_dataset', data=test_data)

            # Matt's Dataset Creation
            # Walking
            # Matt_Group = hdf.create_group('/Matt_Group')
            # Matt_Group.create_dataset('mbrpw', data=MleftBackPocketW)
            # Matt_Group.create_dataset('mrhw', data=MrightHandW)
            # Matt_Group.create_dataset('mrjw', data=MrightJacketPocketW)
            # Matt_Group.create_dataset('mrfpw', data=MrightFrontPocketW)
            # Matt_Group.create_dataset('mljw', data=MleftJacketPocketW)
            # # Jumping
            # Matt_Group.create_dataset('mbrpj', data=MleftBackPocketJ)
            # Matt_Group.create_dataset('mrhj', data=MrightHandJ)
            # Matt_Group.create_dataset('mrjj', data=MrightJacketPocketJ)
            # Matt_Group.create_dataset('mrfpj', data=MrightFrontPocketJ)
            # Matt_Group.create_dataset('mljj', data=MleftJacketPocketJ)

            # Warren's Dataset Creation
            # Walking
            # Warren_Group = hdf.create_group('/Warren_Group')
            # Warren_Group.create_dataset('wbrpw', data=WleftBackPocketW)
            # Warren_Group.create_dataset('wrhw', data=WrightHandW)
            # Warren_Group.create_dataset('wrjw', data=WrightJacketPocketW)
            # Warren_Group.create_dataset('wrfpw', data=WrightFrontPocketW)
            # Warren_Group.create_dataset('wljw', data=WleftJacketPocketW)
            # # Jumping
            # Warren_Group.create_dataset('wbrpj', data=WleftBackPocketJ)
            # Warren_Group.create_dataset('wrhj', data=WrightHandJ)
            # Warren_Group.create_dataset('wrjj', data=WrightJacketPocketJ)
            # Warren_Group.create_dataset('wrfpj', data=WrightFrontPocketJ)
            # Warren_Group.create_dataset('wljj', data=WleftJacketPocketJ)

            # Ellen's Dataset Creation
            # Walking
            # Ellen_Group = hdf.create_group('/Ellen_Group')
            # Ellen_Group.create_dataset('ebrpw', data=EleftBackPocketW)
            # Ellen_Group.create_dataset('erhw', data=ErightHandW)
            # Ellen_Group.create_dataset('erjw', data=ErightJacketPocketW)
            # Ellen_Group.create_dataset('erfpw', data=ErightFrontPocketW)
            # Ellen_Group.create_dataset('eljw', data=EleftJacketPocketW)
            # # Jumping
            # Ellen_Group.create_dataset('ebrpj', data=EleftBackPocketJ)
            # Ellen_Group.create_dataset('erhj', data=ErightHandJ)
            # Ellen_Group.create_dataset('erjj', data=ErightJacketPocketJ)
            # Ellen_Group.create_dataset('erfpj', data=ErightFrontPocketJ)
            # Ellen_Group.create_dataset('eljj', data=EleftJacketPocketJ)


        #---------------------------------------------------------------------------------------------------

        #creating a classifier


        #creating dataframe from the features csv
        dfw = pd.read_csv('Data/Features_Combined.csv')
        dfw2 = pd.read_csv('Data/Input_Features.csv')

        #dropping invalid data entries
        df2 = dfw.dropna()

        inData = dfw2.dropna()

        #splitting the training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(df2[['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance',
                   'Skew', 'Kurtosis', 'Standard Deviation', 'Inter Quartile Range']], Full_set['label'].values[::window_size][:],
                                                                test_size=0.1, shuffle=True)
        #normalizing the input file
        text_input = pd.DataFrame(data=scaler.fit_transform(inData.iloc[:window_size,:]))

        scaler = StandardScaler()
        l_reg = LogisticRegression(max_iter=10000)
        clf = make_pipeline(StandardScaler(), l_reg)
        clf.fit(X_train, Y_train)
        #getting accuracy for the model without applying PCA
        Y_pred = clf.predict(X_test)
        y_clf_prob = clf.predict_proba(X_test)

        #applying PCA to the test data
        normalization = StandardScaler()
        l_reg = LogisticRegression(max_iter=10000)
        pca = PCA(n_components=2)

        pca_pipe = make_pipeline(normalization, pca)

        #applying PCA
        X_train_pca = pca_pipe.fit_transform(X_train)
        X_test_pca = pca_pipe.fit_transform(X_test)

        clf = make_pipeline(l_reg)

        clf.fit(X_train_pca, Y_train)

        y_pred_pca = clf.predict(X_test_pca)

        #Making the Decion boundary Grpah
        # disp = DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, response_method = "predict", xlabel='X1', ylabel='X2', alpha=0.5)
        # disp.ax_.scatter(X_train_pca[:,0], X_train_pca[:,1], c=Y_train)

        #showing accuracy
        acc = accuracy_score(Y_test, y_pred_pca)
        print('accuracy is', acc)

        # creating learning curve
        # train_sizes, train_scores, test_scores = learning_curve(clf, X_train, Y_train, cv=5)
        #
        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        # test_mean = np.mean(test_scores, axis=1)
        # test_std = np.std(test_scores, axis=1)
        #
        # # plotting learning curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
        # plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='red')
        # plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
        # plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        # plt.legend(loc="best")
        # plt.xlabel("Training examples")
        # plt.ylabel("Score")
        # plt.title("Learning curve")
        # plt.show()


        # plt.show()

        #Applying PCA to input file
        pca_pipeIn = make_pipeline(normalization, pca)
        test_pca = pca_pipeIn.fit_transform(text_input)

        text_predict_pca = clf.predict(test_pca)

        #making output file with the labels
        text_input["Action"] = text_predict_pca
        text_input.columns = ['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance',
                   'Skew', 'Kurtosis', 'Standard Deviation', 'Inter Quartile Range', 'Action']

        #saving it to a csv file
        text_input.to_csv('Data/input.csv', index=False)





# GUI

#function for opening a csv file for the input
def open_file():
    """Open a file for editing."""
    label2 = tk.Label(master = form_buttons, text='Proccessing...', fg='red')
    label2.grid(row=7, column=0, sticky='ns')
    filepath = askopenfilename(
        filetypes=[("CSV Files", "*.csv")]
    )
    if not filepath:
        return

    with open(filepath, mode="r", encoding="utf-8") as input_file:
        text = input_file.read()

        #going to the analyzedata function
        analyzedata(text)
        label2.after(100, label2.destroy)
        label3 =tk.Label(master = form_buttons,text = 'Processing Finished!', fg = 'red')
        label3.grid(row=8, column=0, sticky='ns')

        window.title(f"Movement Tracking - {filepath}")
#outputting the graphs + the csv file in the text box
def outputOpen():
    openCSV()
    File = pd.read_csv('Data/input.csv')
    File["Windows"] = File.index

    fig, axs = plt.subplots(figsize=(10, 10))


    axs.scatter(File.iloc[:, 11], File.iloc[:, 10])
    print(File)
    axs.set_title('Output File Action')
    axs.set_xlabel('Window')
    axs.set_ylabel('Acceleration')

    fig.tight_layout()
    plt.show()

#opening the csv file
def openCSV():
    filepath = 'Data/input.csv'
    with open(filepath, mode="r", encoding="utf-8") as input_file:
        text = input_file.read()

        file_display.insert(tk.END, text )

window = tk.Tk()

window.title("Movement Tracking")
file_display = tk.Text(window)

form_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
label1 =tk.Label(master = form_buttons, text = 'Select Input File')

btn_open = tk.Button(form_buttons, text="Open File", command=open_file)
btn_open.grid(row=1, column=0, sticky="n", padx=5, pady=5)

btn_view = tk.Button(form_buttons, text = "View Output Plots", command = outputOpen)
btn_view.grid(row=3, column=0, sticky="n", padx=5, pady=5)
label2 =tk.Label(master = form_buttons, text = 'Legend: \n 0 = Walking \n 1 = Jumping')

form_buttons.grid(row=0, column=0, sticky="nsew")

label1.grid(row = 0, column = 0, sticky = 'n')
label2.grid(row = 5, column = 0, sticky = 'n')

file_display.grid(row=0, column=1, sticky="nsew")
window.mainloop()




