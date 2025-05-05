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
torch.cuda.empty_cache()

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, GraphNorm
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from captum.attr import IntegratedGradients
from torch_geometric.nn import MLP, EdgeConv # Multi-layer Perceptron
from torch.nn import Linear, Dropout
import shap
from topology_identification import Preprocess
from config_file import *
from model import *
from IEEE_datasets.IEEE33 import config_dict as topology_dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#GLOBAL_BRANCH_LIST = [2, 1, 0, 34, 32, 60, 47, 59, 44, 46, 45, 40, 65, 64, 63, 62, 61, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 119, 117, 116, 115, 114, 113, 112, 111, 110, 120, 109, 118, 108, 107, 106, 105, 103, 102, 101, 100, 99, 98, 97, 96, 95, 104, 94, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 68, 67, 66, 43, 42, 93, 58, 57, 56, 54, 53, 52, 51, 50, 49, 48, 82, 69, 70, 41, 81, 80, 79, 78, 77, 71, 72, 31, 74, 73, 29, 76, 75, 33, 25, 37, 36, 35, 23, 30, 9, 39, 38, 55, 22, 28, 27, 26, 21, 24, 20, 17, 16, 14, 19, 18, 13, 15, 4, 12, 3, 10, 6]

#TODO TIaccuracy was set at 95%, while the DSSEaccuracy was set
# at 0.15 for phase angle mean absolute error (MAE) and
# 0.30% for voltage magnitude mean absolute percentage error
# (MAPE).

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

    def execute_rfe_rf_PMU_caseA(self):

        remaining_branches = [i for i in range(NUM_BRANCHES)]
        if dataset == "MESOGEIA":
            remaining_branches = BRANCH_PICK_LIST
        feature_group_dict = {b_i: [  0*NUM_NODES + branch_data[b_i]["sending_node"],
                                      1*NUM_NODES + branch_data[b_i]["sending_node"],
                                      2*NUM_NODES + b_i,
                                      2*NUM_NODES + NUM_BRANCHES + b_i] for b_i in remaining_branches}
        used_branches = []
        num_features = 4

        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)


        while len(remaining_branches) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_branches:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test  = X_test_init[:, used_indices]

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            print(y_test.shape, pred.shape)

            mse = mean_squared_error(y_test, pred)

            print("Test MSE of RF: ", mse)

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
        if dataset == "MESOGEIA":
            remaining_nodes = NODE_PICK_LIST
        feature_group_dict = {b_i: [   0*NUM_NODES + b_i,
                                       1*NUM_NODES + b_i,
                                       2*NUM_NODES + b_i,
                                       3*NUM_NODES + b_i] for b_i in remaining_nodes}
        used_nodes = []
        num_features = 4

        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)

        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test  = X_test_init[:, used_indices]

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            print(y_test.shape, pred.shape)

            mse = mean_squared_error(y_test, pred)

            print("Test MSE of RF: ", mse)
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
        if dataset == "MESOGEIA":
            remaining_nodes = NODE_PICK_LIST
        feature_group_dict = {b_i: [   0*NUM_NODES + b_i,
                                       1*NUM_NODES + b_i,
                                       2*NUM_NODES + b_i] for b_i in remaining_nodes}
        used_nodes = []
        num_features = 3

        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)


        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test  = X_test_init[:, used_indices]

            # Train Random Forest Classifier
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            print(y_test.shape, pred.shape)

            mse = mean_squared_error(y_test, pred)
            print("Test MSE of RF: ", mse)

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
        model.add(Dense(64, input_dim=input_dim, activation='linear'))
        #model.add(Dense(128, activation='linear'))
        model.add(Dense(32, activation='linear'))

        # Output Layer (assuming 16 outputs, modify according to your use case)
        # For regression, we use 'linear' or no activation in the output layer
        model.add(Dense(output_dim, activation='linear'))

        # Compile the model for regression
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Summary of the model
        #model.summary()

        return model

    def build_simple_angles_nn(self, input_dim, output_dim):

        # Define the model
        model = Sequential()

        # Input Layer (66 inputs)
        model.add(Dense(64, input_dim=input_dim, activation='linear'))
        #model.add(Dense(128, activation='linear'))
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

class DSSE_TrainModel:

    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test):

        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test
        print(self.model)


    def train_model(self):
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # You can use 'val_accuracy' or any other metric you are monitoring
            patience=40,  # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore model weights from the epoch with the best metric
        )

        # Train the model and save the history
        history = self.model.fit(self.X_train,
                                 self.y_train,
                                 epochs=300,
                                 batch_size=BATCH_SIZE,
                                 callbacks=[early_stopping],
                                 validation_data=(self.X_val, self.y_val), verbose=1)
        if False:
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

    def __init__(self, meterType, model, X_train, y_train, train_labels, X_val, y_val, val_labels,
                 X_test, y_test, test_labels, old_PMUs=[], FS="RF", method="max", iterative_fs=False):

        self.meterType = meterType
        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test
        self.train_labels, self.val_labels, self.test_labels = train_labels, val_labels, test_labels
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

            #features = FS.execute()
            #features =  []
            if self.meterType == "PMU_caseA":
                #features = IEEE33_PMU_caseA_SE_features
                #features  = MESOGEIA
                features  = UKGD95_PMU_caseA_SE_features
            elif self.meterType == "PMU_caseB":
                #features = IEEE33_PMU_caseB_SE_features
                #features = MESOGEIA_PMU_caseB_SE_features
                features  = UKGD95_PMU_caseB_SE_features
            elif self.meterType == "conventional":
                #features = IEEE33_conventional_SE_features
                #features = MESOGEIA_conventional_SE_features
                features  = UKGD95_conventional_SE_features


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

                    if (mape_magnitudes <= MAPE_v_threshold) and (mae_angles <= MAE_a_threshold): break

            return used_feature_indices

    def execute_GNN(self):

        used_feature_indices = []
        used_branches        = []
        all_indices          = []
        branches = []

        FS = FSPreProc_SE(self.meterType, self.FS, self.method, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, old_PMUs=self.old_PMUs)

        #branches = FS.execute()
        branches = []
        if self.meterType == "PMU_caseA":
            #branches = IEEE33_PMU_caseA_SE_features
            branches  = UKGD95_PMU_caseA_SE_features
        elif self.meterType == "PMU_caseB":
            #branches = IEEE33_PMU_caseB_SE_features
            branches  = UKGD95_PMU_caseB_SE_features
        elif self.meterType == "conventional":
            #branches = IEEE33_conventional_SE_features
            branches  = UKGD95_conventional_SE_features

        #TODO Train for magnitudes
        DSSE_GNN_PP = DSSE_GNN_Preprocess(meterType=self.meterType,
                                          selected_edges=self.old_PMUs,
                                          X_train=self.X_train,
                                          y_train=self.y_train,
                                          train_labels=self.train_labels,
                                          X_val=self.X_val,
                                          y_val=self.y_val,
                                          val_labels=self.val_labels,
                                          X_test=self.X_test,
                                          y_test=self.y_test,
                                          test_labels=self.test_labels)


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

    def __init__(self, meterType, selected_edges, X_train, y_train, train_labels, X_val, y_val, val_labels,
                 X_test, y_test, test_labels):
        self.meterType = meterType
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
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

            #topology = f"T" + str(int(np.argmax(self.train_labels[i]) + 1))
            #open_branches = topology_dict["IEEE33"][topology]["open_branches"]
            #edge_index = self.edge_index[:, [i for i in range(NUM_BRANCHES) if (i not in open_branches)]]
            #edge_mask = torch.tensor([i for i in range(NUM_BRANCHES) if (i in open_branches)])
            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_edge_attr = torch.tensor(val_edge_data[i].reshape(-1, self.num_features_per_edge), dtype=torch.float)
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            #topology = f"T" + str(int(np.argmax(self.train_labels[i]) + 1))
            #open_branches = topology_dict["IEEE33"][topology]["open_branches"]
            #edge_index = self.edge_index[:, [i for i in range(NUM_BRANCHES) if (i not in open_branches)]]
            #edge_mask = torch.tensor([i for i in range(NUM_BRANCHES) if (i in open_branches)])
            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


        test_data = []
        for i in range(self.X_test.shape[0]):
            tmp_edge_attr = torch.tensor(test_edge_data[i].reshape(-1, self.num_features_per_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            #topology = f"T" + str(int(np.argmax(self.train_labels[i]) + 1))
            #open_branches = topology_dict["IEEE33"][topology]["open_branches"]
            #edge_index = self.edge_index[:, [i for i in range(NUM_BRANCHES) if (i not in open_branches)]]
            #edge_mask = torch.tensor([i for i in range(NUM_BRANCHES) if (i in open_branches)])
            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        return self.edge_index, self.train_loader, self.val_loader, self.test_loader

    def generate_dataset_GNN_PMU_caseB(self, output="magnitudes"):
        num_edges = len(branch_data)
        num_nodes = NUM_NODES
        num_features = 4

        print("selected edges: ", self.selected_edges)

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

            train_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            val_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            test_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        return self.edge_index, self.train_loader, self.val_loader, self.test_loader

    def generate_dataset_GNN_conventional(self, output="magnitudes"):
        num_edges = len(branch_data)
        num_nodes = NUM_NODES
        num_features = 3
        edge_index = self.define_graph()

        print(self.selected_edges)

        # self.selected_edges = [6, 32, 9]

        # NP_train, selected_edge_indices, num_edges, num_features_per_edge, num_nodes=33, num_features_per_node=2
        train_node_data, train_node_mask    = self.preprocess_data_conventional(self.X_train, self.selected_edges, num_nodes, num_features)
        val_node_data, val_edge_data        = self.preprocess_data_conventional(self.X_val, self.selected_edges, num_nodes, num_features)
        test_node_data, test_edge_data      = self.preprocess_data_conventional(self.X_test, self.selected_edges, num_nodes, num_features)

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):
            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes": label = torch.tensor(self.y_train[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles": label = torch.tensor(self.y_train[i, NUM_NODES:], dtype=torch.float)

            train_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes":
                label = torch.tensor(self.y_val[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles":
                label = torch.tensor(self.y_val[i, NUM_NODES:], dtype=torch.float)

            val_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, self.num_features_per_node), dtype=torch.float)

            if output == "magnitudes":
                label = torch.tensor(self.y_test[i, :NUM_NODES], dtype=torch.float)
            elif output == "angles":
                label = torch.tensor(self.y_test[i, NUM_NODES:], dtype=torch.float)

            test_data.append(Data(x=tmp_node_attr, edge_index=edge_index, y=label))

        self.test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

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

            #TODO Vanilla GATCONV
            #self.model = SE_GATWithEdgeAttr(num_features=2,output_dim=NUM_NODES,edge_attr_dim=2, gat_layers=4,
            #                                GAT_dim=12, heads=6).to(self.device)

            #TODO Transformer based
            self.model = SE_GATTransfomerOnlyDecoderWithEdges(num_nodes=NUM_NODES, num_features=2,output_dim=NUM_NODES,
                                                              embedding_dim=4, heads=4, num_decoder_layers=1,
                                                              edge_attr_dim=2, gat_layers=2, GATConv_dim=12,
                                                              ff_hid_dim=48).to(self.device)


        elif self.meterType == "PMU_caseB":
            #TODO Vanilla GATConv
            self.model = SE_GATNoEdgeAttrs(num_features=4,output_dim=NUM_NODES, heads=4, gat_layers=2, GAT_dim=8).to(self.device)

            #self.model = SE_GATTransfomerOnlyDecoderNoEdges(num_nodes=NUM_NODES,num_features=4,output_dim=NUM_NODES,embedding_dim=4,
            #                                      heads=4, num_decoder_layers=1,gat_layers=5,GATConv_dim=12,
            #                                      ff_hid_dim=48).to(self.device)
        elif self.meterType == "conventional":
            self.model = SE_GATNoEdgeAttrs(num_features=3,output_dim=NUM_NODES, heads=4, gat_layers=4, GAT_dim=12).to(self.device)
            #self.model = SE_GATTransfomerOnlyDecoderNoEdges(num_nodes=NUM_NODES,num_features=3,output_dim=NUM_NODES,embedding_dim=3,
            #                                      heads=4, num_decoder_layers=1,gat_layers=4,GATConv_dim=12,
            #                                     ff_hid_dim=48).to(self.device)

        print(self.model)
        print("# Trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.output_dim = NUM_NODES

        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5, verbose=True)

    def _train(self):

        # Early stopping parameters
        patience = 40  # Number of epochs to wait for improvement
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

    meterType = "conventional"
    if meterType == "conventional":
        if dataset == "IEEE33":
            old_PMUs = [27, 11, 7, 28, 13, 21, 24, 12, 29, 6, 9, 8, 26, 30, 17, 20, 16, 32, 14, 31, 25, 15, 10]
        elif dataset == "MESOGEIA":
            old_PMUs = [94]
        elif dataset == "95UKGD":
            old_PMUs = [57, 73, 17]
    elif meterType == "PMU_caseB":
        if dataset == "IEEE33":
            old_PMUs = [17, 27]
        elif dataset == "MESOGEIA":
            old_PMUs = [127, 128, 124] #, 123, 127]
        elif dataset == "95UKGD":
            old_PMUs = [79]
    elif meterType == "PMU_caseA":
        if dataset == "IEEE33":
            old_PMUs = [6, 33]
        elif dataset == "MESOGEIA":
            old_PMUs = [130]
        elif dataset == "95UKGD":
            old_PMUs = [75]

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
                                          train_labels = y_train_labels,
                                          X_val=X_val,
                                          y_val=y_val_outputs,
                                          val_labels = y_val_labels,
                                          X_test=X_test,
                                          y_test=y_test_outputs,
                                          test_labels = y_test_labels,
                                          old_PMUs=old_PMUs,
                                          FS="RF",
                                          method="rfe")

    used_features = DSSE_FS.execute()
    print(used_features)
