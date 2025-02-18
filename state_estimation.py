import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
# Evaluate performance on the validation set
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, GINConv, GraphNorm
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from captum.attr import IntegratedGradients
from torch_geometric.nn import MLP, EdgeConv # Multi-layer Perceptron
from torch.nn import Linear, Dropout
import shap
# Set the device globally

#TODO Insert from TI
from topology_identification import Preprocess

#GLOBAL_BRANCH_LIST = [2, 1, 0, 34, 32, 60, 47, 59, 44, 46, 45, 40, 65, 64, 63, 62, 61, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 119, 117, 116, 115, 114, 113, 112, 111, 110, 120, 109, 118, 108, 107, 106, 105, 103, 102, 101, 100, 99, 98, 97, 96, 95, 104, 94, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 68, 67, 66, 43, 42, 93, 58, 57, 56, 54, 53, 52, 51, 50, 49, 48, 82, 69, 70, 41, 81, 80, 79, 78, 77, 71, 72, 31, 74, 73, 29, 76, 75, 33, 25, 37, 36, 35, 23, 30, 9, 39, 38, 55, 22, 28, 27, 26, 21, 24, 20, 17, 16, 14, 19, 18, 13, 15, 4, 12, 3, 10, 6]

#TODO Subclass (or derived class) that extends Animal
# class Dog(Animal):
#    def __init__(self, name, breed):
#        # Call the constructor of the base class
#        super().__init__(name)
#        self.breed = breed
#        # Override the speak method
#    def speak(self):
#        print(f"{self.name} barks")

#TODO TIaccuracy was set at 95%, while the DSSEaccuracy was set
# at 0.15 for phase angle mean absolute error (MAE) and
# 0.30% for voltage magnitude mean absolute percentage error
# (MAPE).

NUM_NODES    = 33
NUM_BRANCHES = 35
NUM_TOPOLOGIES = 15
NUM_SAMPLES = 1000
MAPE_v_threshold = 0.30
MAE_a_threshold  = 0.15

#branch_data = {
#    0: {'sending_node': 0, 'receiving_node': 1},
#    1: {'sending_node': 1, 'receiving_node': 2},
#    2: {'sending_node': 2, 'receiving_node': 3},
#    3: {'sending_node': 3, 'receiving_node': 4},
#    4: {'sending_node': 4, 'receiving_node': 5},
#    5: {'sending_node': 5, 'receiving_node': 6},
#    6: {'sending_node': 6, 'receiving_node': 7},
#    7: {'sending_node': 7, 'receiving_node': 8},
#    8: {'sending_node': 8, 'receiving_node': 9},
#    9: {'sending_node': 9, 'receiving_node': 10},
#    10: {'sending_node': 10, 'receiving_node': 11},
#    11: {'sending_node': 11, 'receiving_node': 12},
#    12: {'sending_node': 12, 'receiving_node': 13},
#    13: {'sending_node': 13, 'receiving_node': 14},
#    14: {'sending_node': 14, 'receiving_node': 15},
#    15: {'sending_node': 15, 'receiving_node': 16},
#    16: {'sending_node': 16, 'receiving_node': 17},
#    17: {'sending_node': 1, 'receiving_node': 18},
#    18: {'sending_node': 18, 'receiving_node': 19},
#    19: {'sending_node': 19, 'receiving_node': 20},
#    20: {'sending_node': 20, 'receiving_node': 21},
#    21: {'sending_node': 2, 'receiving_node': 22},
#    22: {'sending_node': 22, 'receiving_node': 23},
#    23: {'sending_node': 23, 'receiving_node': 24},
#    24: {'sending_node': 5, 'receiving_node': 25},
#    25: {'sending_node': 25, 'receiving_node': 26},
#    26: {'sending_node': 26, 'receiving_node': 27},
#    27: {'sending_node': 27, 'receiving_node': 28},
#    29: {'sending_node': 29, 'receiving_node': 30},
#    30: {'sending_node': 30, 'receiving_node': 31},
#    31: {'sending_node': 31, 'receiving_node': 32},
#    32: {'sending_node': 20, 'receiving_node': 7},
#    33: {'sending_node': 11, 'receiving_node': 21},
#    34: {'sending_node': 24, 'receiving_node': 28}
#}

# MESOGEIA
branch_data = {
    0: {"sending_node": 1, "receiving_node": 0},
    1: {"sending_node": 1, "receiving_node": 2},
    2: {"sending_node": 2, "receiving_node": 3},
    3: {"sending_node": 2, "receiving_node": 4},
    4: {"sending_node": 5, "receiving_node": 3},
    5: {"sending_node": 3, "receiving_node": 6},
    6: {"sending_node": 7, "receiving_node": 5},
    7: {"sending_node": 6, "receiving_node": 8},
    8: {"sending_node": 8, "receiving_node": 9},
    9: {"sending_node": 8, "receiving_node": 10},
    10: {"sending_node": 11, "receiving_node": 10},
    11: {"sending_node": 10, "receiving_node": 12},
    12: {"sending_node": 12, "receiving_node": 13},
    13: {"sending_node": 14, "receiving_node": 11},
    14: {"sending_node": 17, "receiving_node": 6},
    15: {"sending_node": 15, "receiving_node": 12},
    16: {"sending_node": 13, "receiving_node": 16},
    17: {"sending_node": 13, "receiving_node": 18},
    18: {"sending_node": 16, "receiving_node": 19},
    19: {"sending_node": 16, "receiving_node": 21},
    20: {"sending_node": 22, "receiving_node": 19},
    21: {"sending_node": 20, "receiving_node": 17},
    22: {"sending_node": 19, "receiving_node": 23},
    23: {"sending_node": 23, "receiving_node": 24},
    24: {"sending_node": 24, "receiving_node": 25},
    25: {"sending_node": 26, "receiving_node": 23},
    26: {"sending_node": 21, "receiving_node": 27},
    27: {"sending_node": 24, "receiving_node": 28},
    28: {"sending_node": 28, "receiving_node": 29},
    29: {"sending_node": 29, "receiving_node": 30},
    30: {"sending_node": 28, "receiving_node": 31},
    31: {"sending_node": 30, "receiving_node": 32},
    32: {"sending_node": 33, "receiving_node": 29},
    33: {"sending_node": 32, "receiving_node": 34},
    34: {"sending_node": 35, "receiving_node": 30},
    35: {"sending_node": 34, "receiving_node": 37},
    36: {"sending_node": 32, "receiving_node": 36},
    37: {"sending_node": 35, "receiving_node": 40},
    38: {"sending_node": 34, "receiving_node": 38},
    39: {"sending_node": 36, "receiving_node": 39},
    40: {"sending_node": 38, "receiving_node": 47},
    41: {"sending_node": 39, "receiving_node": 41},
    42: {"sending_node": 43, "receiving_node": 39},
    43: {"sending_node": 45, "receiving_node": 35},
    44: {"sending_node": 41, "receiving_node": 42},
    45: {"sending_node": 44, "receiving_node": 45},
    46: {"sending_node": 46, "receiving_node": 43},
    47: {"sending_node": 48, "receiving_node": 46},
    48: {"sending_node": 42, "receiving_node": 49},
    49: {"sending_node": 52, "receiving_node": 48},
    50: {"sending_node": 53, "receiving_node": 46},
    51: {"sending_node": 49, "receiving_node": 50},
    52: {"sending_node": 48, "receiving_node": 51},
    53: {"sending_node": 54, "receiving_node": 44},
    54: {"sending_node": 54, "receiving_node": 44},
    55: {"sending_node": 47, "receiving_node": 55},
    56: {"sending_node": 42, "receiving_node": 59},
    57: {"sending_node": 51, "receiving_node": 56},
    58: {"sending_node": 55, "receiving_node": 62},
    59: {"sending_node": 57, "receiving_node": 49},
    60: {"sending_node": 59, "receiving_node": 58},
    61: {"sending_node": 60, "receiving_node": 52},
    62: {"sending_node": 45, "receiving_node": 61},
    63: {"sending_node": 47, "receiving_node": 65},
    64: {"sending_node": 50, "receiving_node": 63},
    65: {"sending_node": 58, "receiving_node": 64},
    66: {"sending_node": 63, "receiving_node": 66},
    67: {"sending_node": 67, "receiving_node": 50},
    68: {"sending_node": 64, "receiving_node": 82},
    69: {"sending_node": 51, "receiving_node": 70},
    70: {"sending_node": 57, "receiving_node": 68},
    71: {"sending_node": 69, "receiving_node": 57},
    72: {"sending_node": 66, "receiving_node": 74},
    73: {"sending_node": 75, "receiving_node": 66},
    74: {"sending_node": 71, "receiving_node": 56},
    75: {"sending_node": 70, "receiving_node": 72},
    76: {"sending_node": 63, "receiving_node": 79},
    77: {"sending_node": 74, "receiving_node": 73},
    78: {"sending_node": 76, "receiving_node": 67},
    79: {"sending_node": 71, "receiving_node": 83},
    80: {"sending_node": 72, "receiving_node": 84},
    81: {"sending_node": 77, "receiving_node": 54},
    82: {"sending_node": 79, "receiving_node": 78},
    83: {"sending_node": 73, "receiving_node": 85},
    84: {"sending_node": 80, "receiving_node": 67},
    85: {"sending_node": 69, "receiving_node": 81},
    86: {"sending_node": 78, "receiving_node": 87},
    87: {"sending_node": 89, "receiving_node": 69},
    88: {"sending_node": 83, "receiving_node": 86},
    89: {"sending_node": 84, "receiving_node": 91},
    90: {"sending_node": 88, "receiving_node": 74},
    91: {"sending_node": 86, "receiving_node": 90},
    92: {"sending_node": 92, "receiving_node": 71},
    93: {"sending_node": 87, "receiving_node": 98},
    94: {"sending_node": 75, "receiving_node": 93},
    95: {"sending_node": 81, "receiving_node": 94},
    96: {"sending_node": 86, "receiving_node": 95},
    97: {"sending_node": 90, "receiving_node": 96},
    98: {"sending_node": 91, "receiving_node": 97},
    99: {"sending_node": 85, "receiving_node": 99},
    100: {"sending_node": 93, "receiving_node": 100},
    101: {"sending_node": 93, "receiving_node": 104},
    102: {"sending_node": 101, "receiving_node": 97},
    103: {"sending_node": 102, "receiving_node": 87},
    104: {"sending_node": 100, "receiving_node": 103},
    105: {"sending_node": 105, "receiving_node": 91},
    106: {"sending_node": 107, "receiving_node": 103},
    107: {"sending_node": 108, "receiving_node": 101},
    108: {"sending_node": 103, "receiving_node": 106},
    109: {"sending_node": 97, "receiving_node": 111},
    110: {"sending_node": 99, "receiving_node": 112},
    111: {"sending_node": 109, "receiving_node": 99},
    112: {"sending_node": 111, "receiving_node": 110},
    113: {"sending_node": 81, "receiving_node": 113},
    114: {"sending_node": 81, "receiving_node": 113},
    115: {"sending_node": 110, "receiving_node": 114},
    116: {"sending_node": 113, "receiving_node": 117},
    117: {"sending_node": 106, "receiving_node": 115},
    118: {"sending_node": 113, "receiving_node": 119},
    119: {"sending_node": 117, "receiving_node": 116},
    120: {"sending_node": 106, "receiving_node": 118},
    121: {"sending_node": 121, "receiving_node": 107},
    122: {"sending_node": 116, "receiving_node": 122},
    123: {"sending_node": 112, "receiving_node": 120},
    124: {"sending_node": 122, "receiving_node": 123},
    125: {"sending_node": 122, "receiving_node": 124},
    126: {"sending_node": 114, "receiving_node": 125},
    127: {"sending_node": 123, "receiving_node": 128},
    128: {"sending_node": 125, "receiving_node": 126},
    129: {"sending_node": 126, "receiving_node": 129},
    130: {"sending_node": 127, "receiving_node": 120},
    131: {"sending_node": 127, "receiving_node": 128},
    132: {"sending_node": 129, "receiving_node": 130},
}

branch_data = {
    0: {'sending_node': 0, 'receiving_node': 1},
    1: {'sending_node': 1, 'receiving_node': 2},
    2: {'sending_node': 2, 'receiving_node': 3},
    3: {'sending_node': 3, 'receiving_node': 4},
    4: {'sending_node': 4, 'receiving_node': 5},
    5: {'sending_node': 5, 'receiving_node': 6},
    6: {'sending_node': 6, 'receiving_node': 7},
    7: {'sending_node': 7, 'receiving_node': 8},
    8: {'sending_node': 8, 'receiving_node': 9},
    9: {'sending_node': 9, 'receiving_node': 10},
    10: {'sending_node': 10, 'receiving_node': 11},
    11: {'sending_node': 11, 'receiving_node': 12},
    12: {'sending_node': 12, 'receiving_node': 13},
    13: {'sending_node': 13, 'receiving_node': 14},
    14: {'sending_node': 14, 'receiving_node': 15},
    15: {'sending_node': 15, 'receiving_node': 16},
    16: {'sending_node': 16, 'receiving_node': 17},
    17: {'sending_node': 17, 'receiving_node': 18},
    18: {'sending_node': 2, 'receiving_node': 19},
    19: {'sending_node': 19, 'receiving_node': 20},
    20: {'sending_node': 20, 'receiving_node': 21},
    21: {'sending_node': 2, 'receiving_node': 22},
    22: {'sending_node': 22, 'receiving_node': 23},
    23: {'sending_node': 23, 'receiving_node': 24},
    24: {'sending_node': 5, 'receiving_node': 25},
    25: {'sending_node': 25, 'receiving_node': 26},
    26: {'sending_node': 26, 'receiving_node': 27},
    27: {'sending_node': 27, 'receiving_node': 28},
    28: {'sending_node': 28, 'receiving_node': 29},
    29: {'sending_node': 29, 'receiving_node': 30},
    30: {'sending_node': 30, 'receiving_node': 31},
    31: {'sending_node': 31, 'receiving_node': 32},
    32: {'sending_node': 20, 'receiving_node': 7},
    33: {'sending_node': 11, 'receiving_node': 21},
    34: {'sending_node': 24, 'receiving_node': 28}
}

NODE_PICK_LIST   = [4, 7, 9, 14, 15, 17, 18, 20, 21, 22, 25, 26, 27, 31, 33, 37, 40, 45, 53, 59, 60, 62, 68, 70, 75, 76, 77, 79, 80, 82, 83, 85, 88, 89, 90, 92, 94, 95, 96, 98, 102, 104, 105, 107, 108, 109, 111, 112, 114, 115, 117, 118, 119, 121, 124, 127, 128, 129, 130]
BRANCH_PICK_LIST = [key for key in branch_data.keys() if ((branch_data[key]["receiving_node"] in NODE_PICK_LIST) or (branch_data[key]["sending_node"] in NODE_PICK_LIST))]
print(BRANCH_PICK_LIST)

class Preprocess_SE(Preprocess):

    def __init__(self):

        super().__init__()
        self.scaler_m = StandardScaler()
        self.scaler_a = StandardScaler()

    def DSSE_store_data(self):

        df = pd.read_csv(self.dataset_filename)
        df = df[["Vm_m","Va_m", "Ifm_m", "Ifa_m", "Vm_t", "Va_t", "SimNo", "TopoNo"]]
        data = []
        inputs = []
        labels = []

        print(df.columns)
        print(df.corr())

        for topology in range(1, self.topologies + 1):
            for simulation in range(1, self.simulations + 1):

                #TODO Input
                Vm_m = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Vm_m"].values.tolist()[:-2]
                Va_m = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Va_m"].values.tolist()[:-2]
                Ifm_m = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Ifm_m"].values.tolist()
                Ifa_m = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Ifa_m"].values.tolist()

                #TODO Output
                Vm_t = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Vm_t"].values.tolist()[:-2]
                Va_t = df[(df["TopoNo"] == topology) & (df["SimNo"] == simulation)]["Va_t"].values.tolist()[:-2]

                print(topology, simulation)
                #TODO - Choice of a PMU means Choice of Vm, Va, Im, Ia

                data.append([Vm_m + Va_m + Ifm_m + Ifa_m, Vm_t + Va_t])

        for [x, y] in data:
            inputs.append(np.array(x))
            #TODO Assume Currents are measured at branch i: (node j -> node k)
            # Datset is ordered as [Vm, Va, Im, Ia] - [33, 33, 35, 35]
            # For chosen branch:
            #   Get measurement of j        (Voltage Magnitude)
            #   Get measurement of 33 + j   (Voltage Magnitude)
            #   Get measurement of 66 + i   (Current Magnitude)
            #   Get measurement of 100 + i  (Current angle)
            # If a node is already chosen, ignore its importance, since we already use it



            if (len(x) != 136): # 0 -> 0-32, 33 -> 33-65, 66 -> 66-100, 101 -> 101-135
                print("Issue on sample: ", x, y)
            labels.append(y)

        print("Dataset Size", len(inputs), "Input Size: ", len(inputs[0]))
        print("Dataset Size", len(labels))

        print(f"Saving input into 16_topologies_DSSE_input")
        np.save('datasets/16_topologies_DSSE_input.npy', inputs)
        print(f"Saving input into 16_topologies_DSSE_output")
        np.save('datasets/16_topologies_DSSE_output.npy', labels)

    def DSSE_read_data(self):

        print("----------PREPROCESSIING DATASET------------")

        inputs  = np.load('datasets/16_topologies_DSSE_input.npy')
        outputs = np.load('datasets/16_topologies_DSSE_output.npy')

        print("Input size: ", len(inputs), "Sample size: ", len(inputs[0]))
        print("Label size: ", len(outputs))

        # Reshape the labels to a 2D array
        labels_reshaped = outputs.reshape(-1, 1)

        print("Sample")
        print(inputs[0], outputs[0])


        return [inputs, outputs]


    def DSSE_preprocess_data(self):

        inputs, outputs = self.DSSE_read_data()


        # First split: train+validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20,
                                                          random_state=42)  # 0.25 x 0.8 = 0.2

        # Step 1: Initialize the StandardScaler
        scaler = StandardScaler()
        self.scaler = scaler

        # Step 2: Fit the scaler on the training set and transform it
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = X_train

        # Step 3: Apply the same scaler to the test and validation sets (using transform, not fit_transform)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = X_test
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = X_val

        print("Scaler: ", X_train[0], X_train_scaled[0])

        print("Train: ", X_train.shape, y_train.shape)
        print("Validation: ", X_val.shape, y_val.shape)
        print("Test: ", X_test.shape, y_test.shape)


        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

class FSPreProc_SE():

    def __init__(self, meterType, Preproc_model, submethod, X_train, y_train, X_val, y_val, X_test, y_test, old_PMUs):

        self.meterType          = meterType
        self.Preproc_model      = Preproc_model
        self.submethod          = submethod
        self.X_train            = X_train
        self.y_train            = y_train
        self.X_val              = X_val
        self.y_val              = y_val
        self.X_test             = X_test
        self.y_test             = y_test
        self.old_PMUs           = old_PMUs

    def execute_rf_model_magnitude_angles_max(self):

        # branch importances
        branch_importance_relevant_dict = {i: None for i in branch_data.keys()}

        # Train Random Forest Classifier
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        importances = rf.feature_importances_

        # TODO We need to take into consideration the importance per branch and its nodes
        # 33 Vm, 33 Va, 35 Im, 35 Ia
        used_branches = []
        all_indices = [i for i in range(NUM_BRANCHES) if (i not in self.old_PMUs)]
        for _ in range(len(all_indices)):
            importance_pairs = []
            used_sending_nodes = [branch_data[i]["sending_node"] for i in used_branches]
            #print("All indices: ", all_indices)
            #print("Used sending nodes: ", used_sending_nodes)
            remaining_indices = [branch_index for branch_index in all_indices if (branch_index not in used_branches)]
            #print("Remaining indices: ", remaining_indices)

            for branch_index in remaining_indices:
                sending_node_index = branch_data[branch_index]["sending_node"]
                if sending_node_index not in used_sending_nodes:
                    importance_pairs.append(max(importances[sending_node_index],
                                                importances[NUM_NODES + sending_node_index],
                                                importances[66 + branch_index],
                                                importances[101 + branch_index]))
                else:
                    importance_pairs.append(max(importances[66 + branch_index],
                                                importances[101 + branch_index]))

            # Return branch index of max current importance combinations
            max_value = max(importance_pairs)
            max_importances_index = importance_pairs.index(max_value)
            #print("Importance pairs: ", importance_pairs)
            #print("Index of max element in new importance list: ", max_importances_index)
            best_index = remaining_indices[max_importances_index]
            #print("Best index: ", best_index)
            used_branches.append(remaining_indices[importance_pairs.index(max_value)])

        #print(used_branches)

        # return sorted_indices
        return used_branches

    def execute_rf_model_magnitude_angles_sum(self):

        # branch importances
        branch_importance_relevant_dict = {i: None for i in branch_data.keys()}

        # Train Random Forest Classifier
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        importances = rf.feature_importances_

        # TODO We need to take into consideration the importance per branch and its nodes
        # 33 Vm, 33 Va, 35 Im, 35 Ia
        used_branches = []
        all_indices = [i for i in range(35) if (i not in self.old_PMUs)]
        for _ in range(len(all_indices)):
            importance_pairs = []
            used_sending_nodes = [branch_data[i]["sending_node"] for i in used_branches]
            #print("All indices: ", all_indices)
            #print("Used sending nodes: ", used_sending_nodes)
            remaining_indices = [branch_index for branch_index in all_indices if (branch_index not in used_branches)]
            #print("Remaining indices: ", remaining_indices)

            for branch_index in remaining_indices:
                sending_node_index = branch_data[branch_index]["sending_node"]
                if sending_node_index not in used_sending_nodes:
                    importance_pairs.append(sum([importances[sending_node_index],
                                                importances[NUM_NODES + sending_node_index],
                                                importances[66 + branch_index],
                                                importances[101 + branch_index]]))
                else:
                    importance_pairs.append(sum([importances[66 + branch_index],
                                                importances[101 + branch_index]]))

            # Return branch index of max current importance combinations
            max_value = max(importance_pairs)
            max_importances_index = importance_pairs.index(max_value)
            #print("Importance pairs: ", importance_pairs)
            #print("Index of max element in new importance list: ", max_importances_index)
            best_index = remaining_indices[max_importances_index]
            #print("Best index: ", best_index)
            used_branches.append(remaining_indices[importance_pairs.index(max_value)])

        print(used_branches)

        # return sorted_indices
        return used_branches

    def execute_rfe_rf_themis(self):

        remaining_branches = [i for i in range(35)]
        feature_group_dict = {b_i: [   0 + branch_data[b_i]["sending_node"],
                                      NUM_NODES + branch_data[b_i]["sending_node"],
                                      66 + b_i,
                                     101 + b_i] for b_i in remaining_branches}
        used_branches = []

        while len(remaining_branches) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_branches:
                used_indices.extend(feature_group_dict[b_i])

            X_train = self.X_train[:, used_indices]
            #print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42)
            rf.fit(X_train, self.y_train)
            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[i:i+4]) for i in range(len(remaining_branches))]
            #print("Added importances per group: ", group_importance_list)
            min_importance_branch = group_importance_list.index(min(group_importance_list))
            #print("Minimum importance for worst group: ", min(group_importance_list))
            #print("Chose: ", min_importance_branch)
            #print(remaining_branches, min_importance_branch)
            real_branch_index = remaining_branches.pop(min_importance_branch)
            #print("Min importance branch index in remaining list: ", min_importance_branch, "Min real index: ", real_branch_index)
            used_branches.append(real_branch_index)
            #print(used_branches)

        used_branches = reversed(used_branches)
        print("SE Feature Order: ", used_branches)
        ret = []
        #TODO Remove used indices
        for b_i in used_branches:
            if b_i not in self.old_PMUs:
                ret.append(b_i)

        # return sorted_indices
        return ret
    def execute_rfe_rf_PMU_caseA(self):

        remaining_branches = [i for i in range(NUM_BRANCHES)]
        #remaining_branches = BRANCH_PICK_LIST
        feature_group_dict = {b_i: [  0*NUM_NODES + branch_data[b_i]["sending_node"],
                                      1*NUM_NODES + branch_data[b_i]["sending_node"],
                                      2*NUM_NODES + b_i,
                                      2*NUM_NODES + NUM_BRANCHES + b_i] for b_i in remaining_branches}
        used_branches = []
        num_features = 4

        while len(remaining_branches) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_branches:
                used_indices.extend(feature_group_dict[b_i])

            X_train = self.X_train[:, used_indices]
            #print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            print(X_train.shape, self.y_train.shape)
            rf.fit(X_train, self.y_train)
            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_branches))]
            print("Added importances per group: ", group_importance_list)
            min_importance_branch = group_importance_list.index(min(group_importance_list))
            print("Minimum importance for worst group: ", min(group_importance_list))
            print("Chose: ", min_importance_branch)
            print(remaining_branches, min_importance_branch)
            real_branch_index = remaining_branches.pop(min_importance_branch)
            print("Min importance branch index in remaining list: ", min_importance_branch, "Min real index: ", real_branch_index)
            used_branches.append(real_branch_index)
            print(used_branches)

        used_branches = reversed(used_branches)
        print("SE Feature Order: ", used_branches)
        ret = []
        #TODO Remove used indices
        for b_i in used_branches:
            if b_i not in self.old_PMUs:
                ret.append(b_i)

        # return sorted_indices
        return ret
    def execute_rfe_rf_PMU_caseB(self):

        remaining_nodes = [i for i in range(NUM_NODES)]
        #remaining_nodes = NODE_PICK_LIST
        feature_group_dict = {b_i: [   0*NUM_NODES + b_i,
                                       1*NUM_NODES + b_i,
                                       2*NUM_NODES + b_i,
                                       3*NUM_NODES + b_i] for b_i in remaining_nodes}
        used_nodes = []
        num_features = 4

        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = self.X_train[:, used_indices]
            print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            rf.fit(X_train, self.y_train)
            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_nodes))]
            print("Added importances per group: ", group_importance_list)
            min_importance_branch = group_importance_list.index(min(group_importance_list))
            print("Minimum importance for worst group: ", min(group_importance_list))
            print("Chose: ", min_importance_branch)
            print(remaining_nodes, min_importance_branch)
            real_branch_index = remaining_nodes.pop(min_importance_branch)
            print("Min importance branch index in remaining list: ", min_importance_branch, "Min real index: ", real_branch_index)
            used_nodes.append(real_branch_index)
            print(used_nodes)

        used_branches = reversed(used_nodes)
        print("SE Feature Order: ", used_branches)
        ret = []
        #TODO Remove used indices
        for b_i in used_branches:
            if b_i not in self.old_PMUs:
                ret.append(b_i)

        # return sorted_indices
        return ret
    def execute_rfe_rf_conventional(self):

        remaining_nodes = [i for i in range(NUM_NODES)]
        #remaining_nodes = NODE_PICK_LIST
        feature_group_dict = {b_i: [   0*NUM_NODES + b_i,
                                       1*NUM_NODES + b_i,
                                       2*NUM_NODES + b_i] for b_i in remaining_nodes}
        used_nodes = []
        num_features = 3

        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = self.X_train[:, used_indices]
            #print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            rf.fit(X_train, self.y_train)
            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_nodes))]
            print("Added importances per group: ", group_importance_list)
            min_importance_branch = group_importance_list.index(min(group_importance_list))
            print("Minimum importance for worst group: ", min(group_importance_list))
            print("Chose: ", min_importance_branch)
            print(remaining_nodes, min_importance_branch)
            real_branch_index = remaining_nodes.pop(min_importance_branch)
            print("Min importance branch index in remaining list: ", min_importance_branch, "Min real index: ", real_branch_index)
            used_nodes.append(real_branch_index)
            print("Used nodes: ", used_nodes)

        used_branches = reversed(used_nodes)
        print("SE Feature Order: ", used_branches)
        ret = []
        #TODO Remove used indices
        for b_i in used_branches:
            if b_i not in self.old_PMUs:
                ret.append(b_i)

        # return sorted_indices
        return ret

    def execute_pca_model_magnitude_angles(self):

        # Initialize PCA
        pca = PCA(n_components=136)  # Keep all components for now
        X_pca = pca.fit_transform(self.X_train)

        # Explained variance ratio for each component
        explained_variances = pca.explained_variance_ratio_

        loadings = pca.components_

        used_branches = []
        all_indices = [i for i in range(35) if i not in self.old_PMUs]
        for _ in range(len(all_indices)):
            importance_pairs = []
            used_sending_nodes = [branch_data[i]["sending_node"] for i in used_branches]
            #print("All indices: ", all_indices)
            #print("Used sending nodes: ", used_sending_nodes)
            remaining_indices = [branch_index for branch_index in all_indices if (branch_index not in used_branches)]
            #print("Remaining indices: ", remaining_indices)

            for branch_index in remaining_indices:
                tmp_list = []
                sending_node_index = branch_data[branch_index]["sending_node"]
                if sending_node_index not in used_sending_nodes:
                    for i in [sending_node_index, 33 + sending_node_index, 66 + branch_index, 101 + branch_index]:
                        tmp_list.append(i)
                else:
                    for i in [66 + branch_index, 101 + branch_index]:
                        tmp_list.append(i)

                # Calculate the weighted sum for branch importance
                joint_importance = np.sum(np.abs(loadings[:, tmp_list]), axis=1)
                importance_pairs.append(np.sum(joint_importance))

            # Return branch index of max current importance combinations
            max_value = max(importance_pairs)
            max_importances_index = importance_pairs.index(max_value)
            #print("Importance pairs: ", importance_pairs)
            #print("Index of max element in new importance list: ", max_importances_index)
            best_index = remaining_indices[max_importances_index]
            #print("Best index: ", best_index)
            used_branches.append(remaining_indices[importance_pairs.index(max_value)])

        return used_branches

    def execute(self):

        print("------------EXECUTING FEATURE SELECTION-------------")
        print(f"-----------------{self.Preproc_model}----------------------")
        print(f"""------------------{self.submethod}-----------------""")
        if self.Preproc_model == "RF":
            #if self.submethod == "max":
            #    return self.execute_rf_model_magnitude_angles_max()
            #if self.submethod == "sum":
            #    return self.execute_rf_model_magnitude_angles_sum()
            if self.submethod == "rfe":
                if self.meterType == "PMU_caseA":
                    return self.execute_rfe_rf_PMU_caseA()
                elif self.meterType == "PMU_caseB":
                    return self.execute_rfe_rf_PMU_caseB()
                elif self.meterType == "conventional":
                    return self.execute_rfe_rf_conventional()
            else:
                print("Invalid model - submethod combination")
        elif self.Preproc_model == "PCA":
            if self.submethod == "simple":
                return  self.execute_pca_model_magnitude_angles()
            else:
                print("Invalid model - submethod combination")

class DSSE_BuildModel:

    def __init__(self, NN_class, entity):

        self.model=NN_class
        self.entity=entity

    def build_simple_magnitudes_nn(self, input_dim, output_dim):

        # Define the model
        model = Sequential()

        # Input Layer (66 inputs)
        model.add(Dense(512, input_dim=input_dim, activation='linear'))
        model.add(Dense(256, activation='linear'))
        model.add(Dense(128, activation='linear'))
        model.add(Dense(64, activation='linear'))
        model.add(Dense(32, activation='linear'))

        # Output Layer (assuming 16 outputs, modify according to your use case)
        # For regression, we use 'linear' or no activation in the output layer
        model.add(Dense(output_dim, activation='linear'))

        # Compile the model for regression
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Summary of the model
        model.summary()

        return model

    def build_simple_angles_nn(self, input_dim, output_dim):

        # Define the model
        model = Sequential()

        # Input Layer (66 inputs)
        model.add(Dense(512, input_dim=input_dim, activation='linear'))
        model.add(Dense(256, activation='linear'))
        model.add(Dense(128, activation='linear'))
        model.add(Dense(64, activation='linear'))
        model.add(Dense(32, activation='linear'))

        # Output Layer (assuming 16 outputs, modify according to your use case)
        # For regression, we use 'linear' or no activation in the output layer
        model.add(Dense(output_dim, activation='linear'))

        # Compile the model for regression
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Summary of the model
        model.summary()

        return model

    def build_model(self, input_dim, output_dim):

        if self.model=="NN":
            if self.entity == "magnitudes":
                return self.build_simple_magnitudes_nn(input_dim, output_dim)
            elif self.entity == "angles":
                return self.build_simple_angles_nn(input_dim, output_dim)

class GATWithEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, output_dim, edge_attr_dim, heads=4):
        super(GATWithEdgeAttrs, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True, edge_dim=edge_attr_dim)  # First GAT layer with edge features
        self.conv2 = GATConv(64 * heads, 32, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Second GAT layer
        self.conv3 = GATConv(32 * heads, 16, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Third GAT layer
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(16 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1),
                                 dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GAT layer with edge attributes
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x
class GATNoEdgeAttrs(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4):
        super(GATNoEdgeAttrs, self).__init__()

        # Graph Attention layers (GATConv)
        # Here, `edge_attr_dim` is the size of the edge features
        # GAT Layers
        self.conv1 = GATConv(num_features, 16, heads=heads, concat=True)
        self.conv2 = GATConv(16 * heads, 8, heads=heads, concat=True)
        self.conv3 = GATConv(8 * heads, 4, heads=heads, concat=True)
        self.conv4 = GATConv(4 * heads, 2, heads=heads, concat=True)
        # self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True, edge_dim=edge_attr_dim)  # Fourth GAT layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(0.3)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(2 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize with zeros (dummy features)
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default node features (1 feature per node)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer with edge attributes
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer with edge attributes
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GAT layer with edge attributes
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GAT layer with edge attributes
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Fourth GAT layer with edge attributes
        # x = self.conv4(x, edge_index, edge_attr)

        # Global mean pooling: Aggregate node features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer: Output the final classes
        x = self.fc(x)

        return x

class SparseGAT(torch.nn.Module):
    def __init__(self, num_features, output_dim, heads=4, sparse_threshold=1e-5):
        super(SparseGAT, self).__init__()
        self.sparse_threshold = sparse_threshold  # Define threshold for sparsity

        # GAT Layers
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)
        self.conv2 = GATConv(64 * heads, 32, heads=heads, concat=True)
        self.conv3 = GATConv(32 * heads, 16, heads=heads, concat=True)
        self.conv4 = GATConv(16 * heads, 8, heads=heads, concat=True)

        # Dropout layer
        self.dropout = Dropout(0.3)

        # Fully connected classifier
        self.fc = Linear(8 * heads, output_dim)

    def forward(self, data):
        # ---- 1. Handle Missing Features ----
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float, device=data.edge_index.device)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- 2. Create Mask for Significant Nodes ----
        mask = x.abs().sum(dim=1) > self.sparse_threshold  # Find nodes with meaningful values
        filtered_nodes = mask.nonzero().squeeze()  # Get indices of kept nodes

        if filtered_nodes.numel() == 0:
            return torch.zeros((batch.max().item() + 1, self.fc.out_features), device=x.device)

        # ---- 3. Reindex Edge Index ----
        node_map = torch.full((x.size(0),), -1, device=x.device)  # Mapping for new indices
        node_map[filtered_nodes] = torch.arange(filtered_nodes.size(0), device=x.device)  # Assign new indices

        mask_edges = mask[edge_index[0]] & mask[edge_index[1]]  # Keep edges between valid nodes
        edge_index = edge_index[:, mask_edges]  # Apply mask
        edge_index = node_map[edge_index]  # Update indices

        # ---- 4. Apply GAT Layers ----
        x = x[filtered_nodes]  # Select nonzero nodes

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        # ---- 5. Graph-Level Pooling ----
        x = global_mean_pool(x, batch[filtered_nodes])  # Pooling only valid nodes

        # ---- 6. Fully Connected Classifier ----
        x = self.fc(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
class GATTransformer(nn.Module):
    def __init__(self, num_features, output_dim, heads=4, num_transformer_layers=2, num_attention_heads=8):
        super(GATTransformer, self).__init__()

        # Graph Attention Network Layers
        self.conv1 = GATConv(num_features, 8, heads=heads, concat=True)  # First GAT layer
        self.conv2 = GATConv(8 * heads, 4, heads=heads, concat=True)  # Second GAT layer
        #self.conv3 = GATConv(8 * heads, 4, heads=heads, concat=True)  # Third GAT layer
        #self.conv4 = GATConv(4 * heads, 2, heads=heads, concat=True)  # Third GAT layer

        # Transformer Encoder Layer (to process long-range dependencies)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=4 * heads, nhead=num_attention_heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_transformer_layers)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Fully connected output layer for classification/regression
        self.fc = nn.Linear(4 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize sparse tensor
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default dummy node features

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- 1. First GAT Layer ----
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # ---- 2. Second GAT Layer ----
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # ---- 3. Third GAT Layer ----
        #x = self.conv3(x, edge_index)
        #x = F.leaky_relu(x)
        #x = self.dropout(x)

        # ---- 4. Third GAT Layer ----
        #x = self.conv4(x, edge_index)
        #x = F.leaky_relu(x)
        #x = self.dropout(x)


        # ---- 4. Apply Transformer Encoder ----
        # The transformer expects inputs with shape (sequence_length, batch_size, feature_dim)
        # So, we need to reshape the output to match that
        x = x.unsqueeze(0)  # Add a batch dimension at the start (sequence length, batch size, feature dimension)
        x = self.transformer_encoder(x)  # Pass through the transformer encoder
        x = x.squeeze(0)  # Remove the sequence dimension after passing through transformer

        # ---- 5. Global Pooling (Mean) ----
        x = global_mean_pool(x, batch)

        # ---- 6. Fully Connected Layer ----
        x = self.fc(x)

        return x

class GAT_ED_Transformer(nn.Module):
    def __init__(self, num_features, output_dim, heads=4, num_encoder_layers=2, num_decoder_layers=2):
        super(GAT_ED_Transformer, self).__init__()

        # Graph Attention Network Layers (Encoder)
        self.conv1 = GATConv(num_features, 64, heads=heads, concat=True)  # First GAT layer
        self.conv2 = GATConv(64 * heads, 16, heads=heads, concat=True)  # Second GAT layer
        #self.conv3 = GATConv(16 * heads, 8, heads=heads, concat=True)  # Second GAT layer

        # Transformer Encoder Layer (to process long-range dependencies)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=16 * heads, nhead=heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder Layer (to generate output sequence)
        self.transformer_decoder_layer = TransformerDecoderLayer(d_model=16 * heads, nhead=heads)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Fully connected output layer for classification/regression
        self.fc = nn.Linear(16 * heads, output_dim)

    def forward(self, data):
        # If there are no node features, initialize sparse tensor
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Default dummy node features

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- 1. First GAT Layer (Encoder) ----
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # ---- 2. Second GAT Layer (Encoder) ----
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # ---- 3. Second GAT Layer (Encoder) ----
        #x = self.conv3(x, edge_index)
        #x = F.leaky_relu(x)
        #x = self.dropout(x)

        # ---- 3. Apply Transformer Encoder ----
        # The transformer expects inputs with shape (sequence_length, batch_size, feature_dim)
        # So, we need to reshape the output to match that
        x = x.unsqueeze(0)  # Add a batch dimension at the start (sequence length, batch size, feature dimension)

        # Pass through the transformer encoder
        encoder_output = self.transformer_encoder(x)  # Encoder output

        # ---- 4. Apply Transformer Decoder ----
        # The decoder takes the encoder output (context) and generates an output sequence
        # We can pass a target sequence or the encoder output itself if we are doing classification/regression
        decoder_output = self.transformer_decoder(encoder_output,
                                                  encoder_output)  # No target sequence here, self-attention

        # ---- 5. Take Output from Decoder ----
        # We can take the last output from the decoder, or apply pooling
        x = decoder_output.squeeze(0)  # Remove sequence dimension

        x = global_mean_pool(x, batch)

        # ---- 6. Fully Connected Layer ----
        x = self.fc(x)

        return x

class GAT_EED_Transformer(nn.Module):
    def __init__(self, num_nodes, num_features, output_dim, embedding_dim=32, heads=4, num_encoder_layers=2, num_decoder_layers=2):
        super(GAT_EED_Transformer, self).__init__()

        # 🔹 Node Embedding Layer
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # 🔹 Feature Transformation (if needed)
        self.feature_fc = nn.Linear(num_features, embedding_dim) if num_features > 0 else None

        # 🔹 GAT Layers
        self.conv1 = GATConv(embedding_dim, 32, heads=heads, concat=True)
        self.conv2 = GATConv(32 * heads, 16, heads=heads, concat=True)

        # 🔹 Transformer Encoder
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=16 * heads, nhead=heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)

        # 🔹 Transformer Decoder
        self.transformer_decoder_layer = TransformerDecoderLayer(d_model=16 * heads, nhead=heads)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # 🔹 Dropout for Regularization
        self.dropout = nn.Dropout(0.3)

        # 🔹 Fully Connected Output Layer
        self.fc = nn.Linear(16 * heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 🔹 Combine Node Embeddings with Features
        if x is not None and self.feature_fc is not None:
            x = self.feature_fc(x)  # Project features to embedding_dim

        # 🔹 GAT Layers
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # 🔹 Transformer Encoder
        x = x.unsqueeze(0)  # Reshape for transformer
        encoder_output = self.transformer_encoder(x)

        # 🔹 Transformer Decoder
        decoder_output = self.transformer_decoder(encoder_output, encoder_output)
        x = decoder_output.squeeze(0)  # Remove sequence dimension

        # 🔹 Global Pooling
        x = global_mean_pool(x, batch)

        # 🔹 Fully Connected Output Layer
        x = self.fc(x)

        return x

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Pass the input through the layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)  # Output layer (no activation for regression)
        return x

class DSSE_TrainModel:

    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test):

        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self):
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # You can use 'val_accuracy' or any other metric you are monitoring
            patience=50,  # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore model weights from the epoch with the best metric
        )

        # Train the model and save the history
        history = self.model.fit(self.X_train,
                                 self.y_train,
                                 epochs=5,
                                 batch_size=32,
                                 callbacks=[early_stopping],
                                 validation_data=(self.X_val, self.y_val), verbose=0)

        # Plot training & validation accuracy and loss values
        plt.figure(figsize=(14, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'], label='Train Accuracy')
        plt.plot(history.history['val_mae'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(loc='upper left')

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')

        # Show plots
        plt.savefig('DSSE_train_plot.png')

        return self.model

class DSSE_Estimator_TrainProcess:

    def __init__(self, meterType, model, X_train, y_train, X_val, y_val, X_test, y_test, old_PMUs=[], FS="RF", method="max", iterative_fs=False):

        self.meterType = meterType
        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test
        self.FS = FS
        self.method = method
        self.iterative_fs = iterative_fs
        self.old_PMUs = old_PMUs
        self.scaler = StandardScaler()

    def evaluate(self, pred, actual, metric):

        # (3000, 33)
        if metric == "MAPE":

            mape_list = []
            per_node_mape_dict = {i:[] for i in range(NUM_NODES)}

            for i in range(pred.shape[0]):
                a           = actual[i, :]
                p           = pred[i, :]
                length      = len(a)
                tmp_mape_list = [100*(abs(a[j] - p[j])/abs(a[j])) for j in range(length)]
                mape_list   = mape_list + tmp_mape_list
                #for j in range(len(tmp_mape_list)):
                #    per_node_mape_dict[j].append(tmp_mape_list[j])

            #for i in range(33):
            #    per_node_mape_dict[i] = sum(per_node_mape_dict[i]) / len(per_node_mape_dict[i])

            mape_v = sum(mape_list) / len(mape_list)
            print("MAPE_v: ", mape_v)

            return mape_v


        # (3000, 33)
        elif metric == "MAE":

            mae_list = []
            per_node_mae_dict = {i:[] for i in range(NUM_NODES)}

            for i in range(pred.shape[0]):
                a           = actual[i, :]
                p           = pred[i, :]
                length      = len(a)
                tmp_mae_list = [abs(a[j] - p[j]) for j in range(length)]
                mae_list   = mae_list + tmp_mae_list
                #for j in range(len(tmp_mae_list)):
                #    per_node_mae_dict[j].append(tmp_mae_list[j])

            #for i in range(33):
            #    per_node_mae_dict[i] = sum(per_node_mae_dict[i]) / len(per_node_mae_dict[i])

            mae_a = sum(mae_list) / len(mae_list)

            print("MAE_a: ",mae_a)

            return mae_a

    def execute_NN(self):

        if not self.iterative_fs:

            #TODO For every Currenct branch input feature add the magnitude and its angle
            # If the magnitude is at index X, then angle is at index X+35
            used_feature_indices = []
            used_features        = []
            all_indices          = []

            FS = FSPreProc_SE(self.meterType, self.FS, self.method, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.old_PMUs)

            features = FS.execute()
            print(features)
            print("SE Feature Selection Order: ", features)

            for feature in self.old_PMUs:
                if self.meterType == "PMU_caseA":
                    old_branch = feature
                    old_node = branch_data[feature]["sending_node"]
                    used_features.append(feature)
                    for index in [old_node, NUM_NODES + old_node, 2*NUM_NODES + old_branch, 2*NUM_NODES+NUM_BRANCHES + old_branch]:  # Vm, Va, Im, Ia
                        all_indices.append(index)
                    all_indices = list(set(all_indices))
                elif self.meterType == "PMU_caseB":
                    used_features.append(feature)
                    old_node = feature
                    for index in [old_node, NUM_NODES + old_node, 2 * NUM_NODES + old_node, 3 * NUM_NODES + old_node]:  # Vm, Va, Im, Ia
                        all_indices.append(index)
                    all_indices = list(set(all_indices))
                elif self.meterType == "conventional":
                    used_features.append(feature)
                    old_node = feature
                    for index in [old_node, NUM_NODES + old_node, 2 * NUM_NODES + old_node]:  # Vm, Va, Im, Ia
                        all_indices.append(index)
                    all_indices = list(set(all_indices))

            #TODO First iterate over list of SMDs used only for TI
            X_train = self.X_train[:, all_indices]
            X_val = self.X_val[:, all_indices]
            X_test = self.X_test[:, all_indices]
            print(X_train.shape, X_val.shape, X_test.shape)

            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

            y_train_m = self.y_train[:, :NUM_NODES]
            y_val_m = self.y_val[:, :NUM_NODES]
            y_test_m = self.y_test[:, :NUM_NODES]

            print(self.y_train[0], y_train_m[0])

            y_train_a = self.y_train[:, NUM_NODES:]
            y_val_a = self.y_val[:, NUM_NODES:]
            y_test_a = self.y_test[:, NUM_NODES:]

            # TODO Magnitudes
            ML_model = "NN"
            buildModel_m = DSSE_BuildModel(ML_model, "magnitudes")

            input_dim = len(all_indices)
            output_dim = len(y_train_m[0])

            self.model_m = buildModel_m.build_simple_magnitudes_nn(input_dim=input_dim, output_dim=output_dim)

            trainModel = DSSE_TrainModel(self.model_m, X_train, y_train_m, X_val, y_val_m, X_test, y_test_m)
            trainModel.train_model()

            # Evaluate the model on the test data
            y_pred_m = self.model_m.predict(X_test)

            mape_magnitudes = self.evaluate(y_pred_m, y_test_m, "MAPE")

            # TODO Angles
            ML_model = "NN"
            buildModel_a = DSSE_BuildModel(ML_model, "angles")

            input_dim = len(all_indices)
            output_dim = len(y_train_a[0])

            self.model_a = buildModel_a.build_model(input_dim=input_dim, output_dim=output_dim)

            trainModel = DSSE_TrainModel(self.model_a, X_train, y_train_a, X_val, y_val_a, X_test, y_test_a)
            trainModel.train_model()

            # Evaluate the model on the test data
            y_pred_a = self.model_a.predict(X_test)

            mae_angles = self.evaluate(y_pred_a, y_test_a, "MAE")

            print("Used Features: ", used_features, " - MAPE_v: ", str(mape_magnitudes), " - MAE_a: ", str(mae_angles))
            filename = "results/DSSE___" + "MODEL___" + str(ML_model) + "___" + "PREPROCESSING_" + str(FS.Preproc_model) + "___SUBMETHOD_" + str(FS.submethod) + "_results.txt"

            print("MAE_a", mae_angles, "MAPE_v", mape_magnitudes)

            with open(filename, "a") as wf:
                wf.write("Used branches (i-1): " + str(
                    used_features) + f""", #PMUs {str(len(used_features))}""" + ", used indices: " + str(
                    all_indices) + ", MAPE_v: " + str(mape_magnitudes) + ", MAE_a: " + str(mae_angles) + "\n")
                wf.close()

            if not ((mape_magnitudes <= MAPE_v_threshold) and (mae_angles <= MAE_a_threshold)):

                for feature in features:
                    if feature not in self.old_PMUs:
                        used_features.append(feature)
                        if self.meterType == "PMU_caseA":
                            node = branch_data[feature]["sending_node"]
                            print("Choosing Ibranch: ", feature, "and corresponding sending node: ", node)
                            for index in [node, NUM_NODES+node, 2*NUM_NODES+feature, 2*NUM_NODES+NUM_BRANCHES+feature]: # Vm, Va, Im, Ia
                                all_indices.append(index)
                            all_indices = list(set(all_indices))
                        elif self.meterType == "PMU_caseB":
                            node = branch_data[feature]["sending_node"]
                            print("Choosing Ibranch: ", feature, "and corresponding sending node: ", node)
                            for index in [node, NUM_NODES + node, NUM_NODES + feature, 2*NUM_NODES + feature, 3*NUM_NODES + feature]:  # Vm, Va, Iinjm, Iinja
                                all_indices.append(index)
                            all_indices = list(set(all_indices))
                        elif self.meterType == "conventional":
                            node = branch_data[feature]["sending_node"]
                            print("Choosing Ibranch: ", feature, "and corresponding sending node: ", node)
                            for index in [node, 1*NUM_NODES + node, 2*NUM_NODES + feature]:  # Vm, P, Q
                                all_indices.append(index)
                            all_indices = list(set(all_indices))

                        X_train = self.X_train[:, all_indices]
                        X_val   = self.X_val[:, all_indices]
                        X_test  = self.X_test[:, all_indices]

                        X_train = self.scaler.fit_transform(X_train)
                        X_val   = self.scaler.transform(X_val)
                        X_test  = self.scaler.transform(X_test)

                        ML_model = "NN"
                        buildModel_m = DSSE_BuildModel(ML_model, "magnitudes")

                        input_dim  = len(all_indices)
                        output_dim = len(y_train_m[0])

                        self.model_m = buildModel_m.build_simple_magnitudes_nn(input_dim=input_dim, output_dim=output_dim)

                        trainModel = DSSE_TrainModel(self.model_m, X_train, y_train_m, X_val, y_val_m, X_test, y_test_m)
                        trainModel.train_model()

                        # Evaluate the model on the test data
                        y_pred_m = self.model_m.predict(X_test)

                        mape_magnitudes = self.evaluate(y_pred_m, y_test_m, "MAPE")

                        #TODO Angles
                        ML_model = "NN"
                        buildModel_a = DSSE_BuildModel(ML_model, "angles")

                        input_dim = len(all_indices)
                        output_dim = len(y_train_a[0])

                        self.model_a = buildModel_a.build_model(input_dim=input_dim, output_dim=output_dim)

                        trainModel = DSSE_TrainModel(self.model_a, X_train, y_train_a, X_val, y_val_a, X_test, y_test_a)
                        trainModel.train_model()

                        #Evaluate the model on the test data
                        y_pred_a = self.model_a.predict(X_test)

                        mae_angles = self.evaluate(y_pred_a, y_test_a, "MAE")

                        print("Used Features: ", used_features, " - MAPE_v: ", str(mape_magnitudes), " - MAE_a: ", str(mae_angles))
                        filename = "results/DSSE___" + "MODEL___" + str(ML_model) + "___" + "PREPROCESSING_" + str(FS.Preproc_model) + "___SUBMETHOD_" + str(FS.submethod) + "_results.txt"

                        print("MAPE_v", mape_magnitudes, "MAE_a", mae_angles)
                        print("Actual Magnitudes: ", list(y_test_m[0]))
                        print("Predicted Magnitudes: ", list(y_pred_m[0]))
                        print("Actual angles: ", list(y_test_a[0]))
                        print("Predicted angles: ", list(y_pred_a[0]))

                        with open(filename, "a") as wf:
                            wf.write("Used branches (i-1): "+ str(used_features)+ f""", #PMUs {str(len(used_features))}"""+", MAPE_v: "+str(mape_magnitudes) + ", MAE_a: "+str(mae_angles) + "\n")
                            wf.close()

                    else:
                        print(f"""Branch {feature} already in TI PMU set""")
                        mape_magnitudes, mae_angles = 1000, 1000

                    if (mape_magnitudes <= 0.30) and (mae_angles <= 0.15): break

            return used_feature_indices

    def execute_GNN(self):

        used_feature_indices = []
        used_branches        = []
        all_indices          = []

        FS = FSPreProc_SE(self.meterType, self.FS, self.method, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, old_PMUs=self.old_PMUs)

        #TODO branches = FS.execute()
        # Release
        branches = []


        #TODO Train for magnitudes
        DSSE_GNN_PP = DSSE_GNN_Preprocess(meterType=self.meterType,
                                          selected_edges=self.old_PMUs,
                                          X_train=self.X_train,
                                          y_train=self.y_train,
                                          X_val=self.X_val,
                                          y_val=self.y_val,
                                          X_test=self.X_test,
                                          y_test=self.y_test)


        edge_indexes, train_loader, val_loader, test_loader = DSSE_GNN_PP.generate_dataset(output="magnitudes")
        DSSE_GNN_TRAIN = Train_GNN_DSSE(meterType=self.meterType, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        mape_v = DSSE_GNN_TRAIN._evaluate(criterion="MAPE")

        edge_indexes, train_loader, val_loader, test_loader = DSSE_GNN_PP.generate_dataset(output="angles")
        DSSE_GNN_TRAIN = Train_GNN_DSSE(meterType=self.meterType, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        mae_a = DSSE_GNN_TRAIN._evaluate(criterion="MAE")

        print(f"""For old TI PMUs {self.old_PMUs}, we got MAPE_v: {str(mape_v)} and MAE_a {str(mae_a)}""")
        print("Used Features: ", used_branches, " - MAPE_v: ", str(mape_v), " - MAE_a: ", str(mae_a))
        filename = "results/DSSE___" + "MODEL___GNN___" + "PREPROCESSING_" + str(FS.Preproc_model) + "___SUBMETHOD_" + str(FS.submethod) + "_results.txt"

        with open(filename, "a") as wf:
            wf.write("Used branches (i-1): " + str(self.old_PMUs) + f""", #PMUs {str(len(self.old_PMUs))}""" + ", MAPE_v: " + str(mape_v) + ", MAE_a: " + str(mae_a) + "\n")
            wf.close()

        if ((mape_v > MAPE_v_threshold) or (mae_a > MAE_a_threshold)):

            new_PMUs = self.old_PMUs

            for branch in branches:
                print("Choosing branch: ", branch)
                if branch not in self.old_PMUs:
                    new_PMUs.append(branch)
                    print("New chosen PMUs: ", new_PMUs)
                    # TODO Train for magnitudes
                    DSSE_GNN_PP = DSSE_GNN_Preprocess(meterType=self.meterType,
                                                      selected_edges=new_PMUs,
                                                      X_train=self.X_train,
                                                      y_train=self.y_train,
                                                      X_val=self.X_val,
                                                      y_val=self.y_val,
                                                      X_test=self.X_test,
                                                      y_test=self.y_test)

                    edge_indexes, train_loader, val_loader, test_loader = DSSE_GNN_PP.generate_dataset(output="magnitudes")
                    DSSE_GNN_TRAIN = Train_GNN_DSSE(meterType=self.meterType, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
                    mape_v = DSSE_GNN_TRAIN._evaluate(criterion="MAPE")

                    edge_indexes, train_loader, val_loader, test_loader = DSSE_GNN_PP.generate_dataset(output="angles")
                    DSSE_GNN_TRAIN = Train_GNN_DSSE(meterType=self.meterType, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
                    mae_a = DSSE_GNN_TRAIN._evaluate(criterion="MAE")

                    print(f"""For new PMUs {new_PMUs}, we got MAPE_v: {str(mape_v)} and MAE_a {str(mae_a)}""")
                    print("Used branches: ", new_PMUs, " - MAPE_v: ", str(mape_v), " - MAE_a: ", str(mae_a))
                    filename = "results/DSSE___" + "MODEL___GNN___" + "PREPROCESSING_" + str(FS.Preproc_model) + "___SUBMETHOD_" + str(FS.submethod) + "_results.txt"

                    with open(filename, "a") as wf:
                        wf.write("Used branches (i-1): " + str(new_PMUs) + f""", #PMUs {str(len(new_PMUs))}""" + ", MAPE_v: " + str(mape_v) + ", MAE_a: " + str(mae_a) + "\n")
                        wf.close()

                    if ((mape_v <= MAPE_v_threshold) and (mae_a <= MAE_a_threshold)):
                        return new_PMUs

            return new_PMUs


    def execute(self):

        if self.model == "NN":
            self.execute_NN()
        elif self.model == "GNN":
            self.execute_GNN()

class DSSE_GNN_Preprocess:

    def __init__(self, meterType, selected_edges, X_train, y_train, X_val, y_val, X_test, y_test):
        self.meterType = meterType
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.selected_edges = selected_edges
        if self.meterType == "PMU_caseA":
            self.num_features_per_edge = 2
            self.num_features_per_node = 2
        elif self.meterType == "PMU_caseB":
            self.num_features_per_edge = 0
            self.num_features_per_node = 4
        elif self.meterType == "conventional":
            self.num_features_per_edge = 0
            self.num_features_per_node = 3

    def define_graph(self):
        # Complete edge_index for all branches (35 edges in total)
        edges = [(v['sending_node'], v['receiving_node']) for v in branch_data.values()]
        # edges = [edges[i] for i in self.selected_edges]
        # print("Selected edges: ", selected_edges)
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Ensure correct shape [2, num_edges]
        return self.edge_index

    def preprocess_data_PMU_caseA(self, NP_train, selected_edge_indices, num_edges, num_features_per_edge, num_nodes=33, num_features_per_node=2):
        # Edges start at index 0 - 32, 33 - 65, 66 - 100, 101 - 135

        branch_index_offset = 2*NUM_NODES
        node_index_offset = 0

        edge_features = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))
        edge_mask = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))

        node_features = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))
        node_mask = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))

        # Only the features for selected edges will be non-zero
        for idx, edge_idx in enumerate(selected_edge_indices):
            mag_feature = NP_train[:, branch_index_offset + edge_idx]
            angle_feature = NP_train[:, branch_index_offset + edge_idx + NUM_BRANCHES]
            edge_features[:, edge_idx * 2] = mag_feature
            edge_features[:, edge_idx * 2 + 1] = angle_feature
            edge_mask[:, edge_idx * 2] = 1
            edge_mask[:, edge_idx * 2 + 1] = 1

            node_idx = branch_data[edge_idx]["sending_node"]
            mag_feature = NP_train[:, node_index_offset + node_idx]
            angle_feature = NP_train[:, node_index_offset + node_idx + NUM_NODES]
            node_features[:, node_idx * 2] = mag_feature
            node_features[:, node_idx * 2 + 1] = angle_feature
            node_mask[:, node_idx * 2] = 1
            node_mask[:, node_idx * 2 + 1] = 1

        return node_features, edge_features

    def preprocess_data_PMU_caseB(self, NP_train, selected_node_indices, num_nodes=NUM_NODES, num_features_per_node=4):

        # Edges start at index Vm: 0 - 32, Va: 33 - 65, Im: 66 - 100, Ia: 101 - 135

        node_features           = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))
        node_mask               = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))

        # Only the features for selected edges will be non-zero
        for idx, node_idx in enumerate(selected_node_indices):
            Vmag_feature    = NP_train[:, 0*NUM_NODES + node_idx]
            Vangle_feature  = NP_train[:, 1*NUM_NODES + node_idx]
            Iinjm_feature   = NP_train[:, 2*NUM_NODES + node_idx]
            Iinja_feature   = NP_train[:, 3*NUM_NODES + node_idx]

            node_features[:, node_idx * num_features_per_node + 0] = Vmag_feature
            node_features[:, node_idx * num_features_per_node + 1] = Vangle_feature
            node_features[:, node_idx * num_features_per_node + 2] = Iinjm_feature
            node_features[:, node_idx * num_features_per_node + 3] = Iinja_feature

            node_mask[:, node_idx * num_features_per_node + 0] = 1
            node_mask[:, node_idx * num_features_per_node + 1] = 1
            node_mask[:, node_idx * num_features_per_node + 2] = 1
            node_mask[:, node_idx * num_features_per_node + 3] = 1

        return node_features, node_mask

    def preprocess_data_conventional(self, NP_train, selected_node_indices, num_nodes=NUM_NODES, num_features_per_node=3):

        # Edges start at index Vm: 0 - 32, Va: 33 - 65, Im: 66 - 100, Ia: 101 - 135

        print("SELECTED NODE INDICES FOR PREPROCESS: ", selected_node_indices)

        node_index_offset       = 0
        node_features           = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))
        node_mask               = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))

        # Only the features for selected edges will be non-zero
        for idx, node_idx in enumerate(selected_node_indices):
            Vmag_feature    = NP_train[:, node_index_offset + node_idx + 0*NUM_NODES]
            P_feature  = NP_train[:, node_index_offset + node_idx + 1*NUM_NODES]
            Q_feature   = NP_train[:, node_index_offset + node_idx + 2*NUM_NODES]

            node_features[:, node_idx * num_features_per_node + 0] = Vmag_feature
            node_features[:, node_idx * num_features_per_node + 1] = P_feature
            node_features[:, node_idx * num_features_per_node + 2] = Q_feature

            node_mask[:, node_idx * num_features_per_node + 0] = 1
            node_mask[:, node_idx * num_features_per_node + 1] = 1
            node_mask[:, node_idx * num_features_per_node + 2] = 1

        return node_features, node_mask

    def generate_dataset_GNN_PMU_caseA(self, output="magnitudes"):
        num_edges = len(branch_data)
        num_nodes = NUM_NODES
        num_features = 2

        print(self.selected_edges)

        # self.selected_edges = [6, 32, 9]

        # NP_train, selected_edge_indices, num_edges, num_features_per_edge, num_nodes=33, num_features_per_node=2
        train_node_data, train_edge_data = self.preprocess_data_PMU_caseA(self.X_train, self.selected_edges, num_edges, num_features, num_nodes, num_features)
        val_node_data, val_edge_data = self.preprocess_data_PMU_caseA(self.X_val, self.selected_edges, num_edges, num_features, num_nodes, num_features)
        test_node_data, test_edge_data = self.preprocess_data_PMU_caseA(self.X_test, self.selected_edges, num_edges, num_features, num_nodes, num_features)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):
            tmp_edge_attr = torch.tensor(train_edge_data[i].reshape(-1, self.num_features_per_edge), dtype=torch.float)
            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output=="magnitudes": label = torch.tensor(self.y_train[i, :NUM_NODES], dtype=torch.float)
            elif output=="angles": label = torch.tensor(self.y_train[i, NUM_NODES:], dtype=torch.float)

            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)


        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_edge_attr = torch.tensor(val_edge_data[i].reshape(-1, self.num_features_per_edge), dtype=torch.float)
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=True)


        test_data = []
        for i in range(self.X_test.shape[0]):
            tmp_edge_attr = torch.tensor(test_edge_data[i].reshape(-1, self.num_features_per_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        return self.edge_index, self.train_loader, self.val_loader, self.test_loader

    def generate_dataset_GNN_PMU_caseB(self, output="magnitudes"):
        num_edges = len(branch_data)
        num_nodes = NUM_NODES
        num_features = 4

        print(self.selected_edges)

        # self.selected_edges = [6, 32, 9]

        # NP_train, selected_edge_indices, num_edges, num_features_per_edge, num_nodes=33, num_features_per_node=2
        train_node_data, train_node_mask        = self.preprocess_data_PMU_caseB(self.X_train, self.selected_edges, num_nodes, num_features)
        val_node_data, val_edge_data            = self.preprocess_data_PMU_caseB(self.X_val, self.selected_edges, num_nodes, num_features)
        test_node_data, test_edge_data          = self.preprocess_data_PMU_caseB(self.X_test, self.selected_edges, num_nodes, num_features)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):
            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_train[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_train[i, NUM_NODES:], dtype=torch.float)

            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        return self.edge_index, self.train_loader, self.val_loader, self.test_loader

    def generate_dataset_GNN_conventional(self, output="magnitudes"):
        num_edges = len(branch_data)
        num_nodes = NUM_NODES
        num_features = 3

        print(self.selected_edges)

        # self.selected_edges = [6, 32, 9]

        # NP_train, selected_edge_indices, num_edges, num_features_per_edge, num_nodes=33, num_features_per_node=2
        train_node_data, train_node_mask    = self.preprocess_data_conventional(self.X_train, self.selected_edges, num_nodes, num_features)
        val_node_data, val_edge_data        = self.preprocess_data_conventional(self.X_val, self.selected_edges, num_nodes, num_features)
        test_node_data, test_edge_data      = self.preprocess_data_conventional(self.X_test, self.selected_edges, num_nodes, num_features)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):
            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes":
                label = torch.tensor(self.y_train[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles":
                label = torch.tensor(self.y_train[i, NUM_NODES:], dtype=torch.float)

            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes":
                label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles":
                label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes":
                label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles":
                label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        return self.edge_index, self.train_loader, self.val_loader, self.test_loader

    def generate_dataset(self, output="magnitudes"):

        if self.meterType == "PMU_caseA":
            edge_index, train_loader, val_loader, test_loader = self.generate_dataset_GNN_PMU_caseA(output="magnitudes")
        elif self.meterType == "PMU_caseB":
            edge_index, train_loader, val_loader, test_loader = self.generate_dataset_GNN_PMU_caseB(output="magnitudes")
        elif self.meterType == "conventional":
            edge_index, train_loader, val_loader, test_loader = self.generate_dataset_GNN_conventional(output="magnitudes")

        return edge_index, train_loader, val_loader, test_loader



class Train_GNN_DSSE:

    def __init__(self, meterType, train_loader, val_loader, test_loader):
        self.meterType = meterType
        self.device = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        if self.meterType == "PMU_caseA":
            self.model = GATWithEdgeAttrs(num_features=2,output_dim=NUM_NODES,edge_attr_dim=2, heads=8).to(self.device)
        elif self.meterType == "PMU_caseB":
            self.model = GATNoEdgeAttrs(num_features=4,output_dim=NUM_NODES, heads=16).to(self.device)
            #self.model = SparseGAT(num_features=4,output_dim=NUM_NODES, heads=8).to(self.device)
            #self.model = GATTransformer(num_features=4, output_dim=NUM_NODES, heads=16, num_transformer_layers=2, num_attention_heads=16).to(self.device)
            #self.model = GAT_ED_Transformer(num_features=4,output_dim=NUM_NODES,heads=8,num_encoder_layers=2,num_decoder_layers=2).to(self.device)
            #self.model = GAT_EED_Transformer(num_nodes=NUM_NODES,num_features=4,output_dim=NUM_NODES,embedding_dim=12,heads=16,num_encoder_layers=2, num_decoder_layers=2).to(self.device)
            print(self.model.parameters())
        elif self.meterType == "conventional":
            self.model = GATNoEdgeAttrs(num_features=3,output_dim=NUM_NODES, heads=8).to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.output_dim = NUM_NODES

        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5, verbose=True)

    def _train(self):

        # Early stopping parameters
        patience = 80  # Number of epochs to wait for improvement
        min_delta = 0.00000001  # Minimum change in validation loss to qualify as an improvement
        best_val_loss = float('inf')
        early_stop_counter = 0
        max_epochs = 1000  # Maximum number of epochs to train
        best_model_weights = None  # To store the best weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)

                out_flat = out.view(-1, self.output_dim)
                y_flat = batch.y.view(-1, self.output_dim)

                #print(batch.x.shape)
                #print(out_flat.shape)
                #print(y_flat.shape)

                loss = self.criterion(out_flat, y_flat)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = 0
            for batch_val in self.val_loader:
                batch_val = batch_val.to(self.device)
                out_val = self.model(batch_val)

                out_flat = out_val.view(-1, self.output_dim)
                y_flat = batch_val.y.view(-1, self.output_dim)

                loss = self.criterion(out_flat, y_flat)
                val_loss += loss.item()

            ## Early stopping and best weights check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0  # Reset counter if validation loss improves
                best_model_weights = self.model.state_dict()  # Save the best weights
            else:
                early_stop_counter += 1

            # Reduce learning rate if validation loss plateaus
            self.scheduler.step(val_loss)

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break  # Stop training if no improvement for `patience` epochs

            print(f'Epoch {epoch + 1}, - Training Loss: {total_loss / len(self.train_loader)}',
              f""" - Validation Loss: {val_loss / len(self.val_loader)}""")

        return self.model.load_state_dict(best_model_weights)

    def _evaluate(self, criterion="MAPE"):

        self._train()
        test_loss = 0
        mape_list = []
        mae_list  = []
        mae, mape = 0, 0

        for batch_test in self.test_loader:
            batch_test = batch_test.to(self.device)
            out_test = self.model(batch_test)

            out_flat = out_test.view(-1, self.output_dim)
            y_flat = batch_test.y.view(-1, self.output_dim)

            loss = self.criterion(out_flat, y_flat)
            test_loss += loss.item()

            if criterion == "MAPE":
                for i in range(out_flat.shape[0]):
                    p = list(out_flat[i, :])
                    p = [i.item() for i in p]
                    a = list(y_flat[i, :])
                    a = [i.item() for i in a]
                    length = len(a)
                    tmp_mape_list = [100 * (abs(a[j] - p[j]) / abs(a[j])) for j in range(length)]
                    mape_list = mape_list + tmp_mape_list
                mape = sum(mape_list)/len(mape_list)

            elif criterion == "MAE":
                for i in range(out_flat.shape[0]):
                    p = list(out_flat[i, :])
                    p = [i.item() for i in p]
                    a = list(y_flat[i, :])
                    a = [i.item() for i in a]
                    length = len(a)
                    tmp_mae_list = [abs(a[j] - p[j]) for j in range(length)]
                    mae_list = mae_list + tmp_mae_list
                mae = sum(mae_list)/len(mae_list)

        print(f""" - Evaluation (Test Set) Loss: {test_loss / len(self.test_loader)}""")

        if criterion == "MAPE":
            print("MAPE_v: ", mape)
            return mape
        elif criterion == "MAE":
            print("MAE_a: ", mae)
            return mae


if __name__ == "__main__":

    meterType = "PMU_caseB"
    if meterType == "conventional":
        old_PMUs = [27, 13] #[124, 127, 128]
    elif meterType == "PMU_caseB":
        old_PMUs = [17, 26]
    elif meterType == "PMU_caseA":
        old_PMUs = [6, 10] #[127, 123]

    model    = "GNN"
    PP       = "RF"
    subPP    = "rfe"

    PP_SE = Preprocess()
    X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, y_test_labels = PP_SE.preprocess_meter_type(meterType)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train_outputs.shape)
    print("X_val   shape: ", X_val.shape)
    print("y_val   shape: ", y_val_outputs.shape)
    print("X_test  shape: ", X_test.shape)
    print("y_test  shape: ", y_test_outputs.shape)

    #FS_PP_SE = FSPreProc_SE(meterType,"RF","rfe",X_train,y_train_outputs,X_val,y_val_outputs,X_test,y_test_outputs,old_PMUs)
    #features = FS_PP_SE.execute()
    #print(features)
    DSSE_FS = DSSE_Estimator_TrainProcess(meterType=meterType,
                                          model=model,
                                          X_train=X_train,
                                          y_train=y_train_outputs,
                                          X_val=X_val,
                                          y_val=y_val_outputs,
                                          X_test=X_test,
                                          y_test=y_test_outputs,
                                          old_PMUs=old_PMUs,
                                          FS="RF",
                                          method="rfe")

    used_features = DSSE_FS.execute()
    print(used_features)