import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
# Evaluate performance on the validation set
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv, GATConv, GATv2Conv, SAGEConv, APPNP, GINConv, GraphNorm
from torch_geometric.data import Data, DataLoader

from torch_geometric.utils import degree
from captum.attr import IntegratedGradients
from torch_geometric.nn import MLP, EdgeConv # Multi-layer Perceptron
from torch.nn import Linear, BatchNorm1d
import shap
from config_file import *
from model import (TI_SimpleNNEdges, TI_GATWithEdgeAttrs, TI_GATNoEdgeAttrs, TI_TransformerNoEdges,
                   TI_GCNNoEdgeAttrs,  TI_GCNNoEdgeAttrs, TI_TransformerWithEdges)
from torch.utils.data import DataLoader as DL_NN, TensorDataset as TD_NN
from IEEE_datasets.IEEE33 import config_dict
from outlier_classes import OutlierInjector, IF_OutlierDetection, KNNImputerMeasurements
import joblib
from tensorflow.keras.callbacks import EarlyStopping


# Set the device globally
np.set_printoptions(threshold=np.inf)

class Preprocess:

    def __init__(self):

        self.filename                  = RAW_FILENAME
        self.topologies                = NUM_TOPOLOGIES
        self.simulations               = NUM_SIMULATIONS
        self.dataset_filename          = RAW_FILENAME

    def store_data_PMU_caseA(self):

        df = pd.read_csv(self.dataset_filename)
        # "Vm_m","Va_m", "Ifm_m", "Ifa_m", "Vm_t", "Va_t", "SimNo", "TopoNo"
        df = df[["vm_pu","va_degree","P_pu","Q_pu","Im_inj","Ia_inj","Im_pu","Ia_pu","TopNo","Simulation"]]
        data = []
        inputs = []
        labels = []

        for topology in range(1, self.topologies + 1):
            for simulation in range(1, self.simulations + 1):
                print("Topology: ", topology, "Simulation: ", simulation)
                # TODO Input
                Vm_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                #Vm_m *= (1 + (additional_error / 3) * np.random.randn(len(Vm_m)))
                #Vm_m = list(Vm_m)

                Va_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["va_degree"].values.tolist()[:-2]
                #Va_m *= (1 + (additional_error / 3) * np.random.randn(len(Va_m)))
                #Va_m = list(Va_m)

                Ifm_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["Im_pu"].values.tolist()
                #Ifm_m *= (1 + (additional_error / 3) * np.random.randn(len(Ifm_m)))
                #Ifm_m = list(Ifm_m)

                Ifa_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["Ia_pu"].values.tolist()
                #Ifa_m *= (1 + (additional_error / 3) * np.random.randn(len(Ifa_m)))
                #Ifa_m = list(Ifa_m)

                # TODO Output
                Vm_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                Va_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["va_degree"].values.tolist()[:-2]

                # Input, SE Output, TI Output
                data.append([Vm_m + Va_m + Ifm_m + Ifa_m, Vm_t + Va_t, topology])
                # data.append([If_real, topology])

        for [x, y, z] in data:
            if len(x) != 2*NUM_NODES+2*NUM_BRANCHES:
                print("Issue on sample: ", x, y)
            else:
                inputs.append(np.array(x))
                y = np.array(y)
                z = np.array([z])
                label = np.concatenate([y, z])
                labels.append(label)

        print(f"Saving input into {PMU_caseA_input}")
        np.save(f"{PMU_caseA_input}", inputs)
        print(f"Saving input into {PMU_caseA_output}")
        np.save(f"{PMU_caseA_output}", labels)

        self.train_test_split_dataset("PMU_caseA")

    def store_data_PMU_caseB(self):

        df = pd.read_csv(self.dataset_filename)
        print(df.shape)
        print("Entering case B")
        # "Vm_m","Va_m", "Ifm_m", "Ifa_m", "Vm_t", "Va_t", "SimNo", "TopoNo"
        df = df[["vm_pu","va_degree","P_pu","Q_pu","Im_inj","Ia_inj","Im_pu","Ia_pu","TopNo","Simulation"]]
        data = []
        inputs = []
        labels = []

        for topology in range(1, self.topologies + 1):
            for simulation in range(1, self.simulations + 1):
                print(topology, simulation)
                # TODO Input
                Vm_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                Va_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["va_degree"].values.tolist()[:-2]
                Iinjm = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["Im_inj"].values.tolist()[:-2]
                Iinja = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["Ia_inj"].values.tolist()[:-2]

                # TODO Output
                Vm_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                Va_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["va_degree"].values.tolist()[:-2]

                # Input, SE Output, TI Output
                data.append([Vm_m + Va_m + Iinjm + Iinja, Vm_t + Va_t, topology])
                # data.append([If_real, topology])

        for [x, y, z] in data:
            if len(x) != 4*NUM_NODES:
                print("Issue on sample: ", x, y)
            else:
                inputs.append(np.array(x))
                y = np.array(y)
                z = np.array([z])
                label = np.concatenate([y, z])
                labels.append(label)

        print("Dataset Size", len(inputs), "Input Size: ", len(inputs[0]))
        print("Dataset Size", len(labels))

        print(f"Saving input into datasets " + PMU_caseB_input)
        np.save(PMU_caseB_input, inputs)
        print(f"Saving input into " + PMU_caseB_output)
        np.save(PMU_caseB_output, labels)

        self.train_test_split_dataset(meterType)


    def store_data_conventional(self):

        #Large error for conventional meters - 1%
        error = 0.009

        df = pd.read_csv(self.dataset_filename)
        # "Vm_m","Va_m", "Ifm_m", "Ifa_m", "Vm_t", "Va_t", "SimNo", "TopoNo"
        df = df[["vm_pu","va_degree","P_pu","Q_pu","Im_inj","Ia_inj","Im_pu","Ia_pu","TopNo","Simulation"]]
        data = []
        inputs = []
        labels = []
        for topology in range(1, self.topologies + 1):
            for simulation in range(1, self.simulations + 1):
                Vm_m = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                Vm_m *= (1 + ((error / 2) / 3) * np.random.randn(len(Vm_m)))
                Vm_m = list(Vm_m)
                P_pu = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["P_pu"].values.tolist()[:-2]
                P_pu *= (1 + (error / 3) * np.random.randn(len(P_pu)))
                P_pu = list(P_pu)
                Q_pu = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["Q_pu"].values.tolist()[:-2]
                Q_pu *= (1 + (error / 3) * np.random.randn(len(Q_pu)))
                Q_pu = list(Q_pu)
                # TODO Input


                print(f"Inserted {110*error}% noise to conventional meters")

                # TODO Output
                Vm_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["vm_pu"].values.tolist()[:-2]
                Va_t = df[(df["TopNo"] == topology) & (df["Simulation"] == simulation)]["va_degree"].values.tolist()[:-2]

                print(topology, simulation)

                # Input, SE Output, TI Output
                data.append([Vm_m + P_pu + Q_pu, Vm_t + Va_t, topology])
                # data.append([If_real, topology])

        for [x, y, z] in data:
            if len(x) != 3 * NUM_NODES:
                print("Issue on sample: ", x, y)
            else:
                inputs.append(np.array(x))
                y = np.array(y)
                z = np.array([z])
                label = np.concatenate([y, z])
                labels.append(label)

        print("Dataset Size", len(inputs), "Input Size: ", len(inputs[0]))
        print("Dataset Size", len(labels))

        print(f"Saving input into ", conventional_input)
        np.save(f"{conventional_input}", inputs)
        print(f"Saving input into {conventional_output}")
        np.save(f"{conventional_output}", labels)

        self.train_test_split_dataset(meterType)


    def custom_one_hot_encode(self, labels):

        one_hot_encoded_labels = []
        for label in labels:
            tmp = [0 for i in range(self.topologies)]
            tmp[int(label)-1] = 1

            one_hot_encoded_labels.append(tmp)

        one_hot_encoded_labels = np.array(one_hot_encoded_labels)

        return one_hot_encoded_labels


    def read_data_PMU_caseA(self):

        print("----------PREPROCESSIING DATASET------------")

        inputs = np.load(PMU_caseA_input)
        outputs = np.load(PMU_caseA_output)

        # Reshape the labels to a 2D array
        outputs, labels = outputs[:, :-1], outputs[:, -1]

        labels_reshaped = list(labels)

        ohe_labels = self.custom_one_hot_encode(labels_reshaped)

        # Initialize an empty list to store concatenated elements
        concatenated_list = []

        # Loop through each element (i) of outputs and labels
        for i in range(len(outputs)):
            # Concatenate the ith elements of outputs and labels
            concatenated = np.concatenate((outputs[i], ohe_labels[i]))
            concatenated_list.append(concatenated)

        concatenated = np.array(concatenated_list)

        return [inputs, concatenated]

    def read_data_PMU_caseB(self):

        print("----------PREPROCESSIING DATASET------------")

        inputs = np.load(PMU_caseB_input)
        outputs = np.load(PMU_caseB_output)

        # Reshape the labels to a 2D array
        outputs, labels = outputs[:, :-1], outputs[:, -1]

        labels_reshaped = list(labels)

        ohe_labels = self.custom_one_hot_encode(labels_reshaped)

        # Initialize an empty list to store concatenated elements
        concatenated_list = []

        # Loop through each element (i) of outputs and labels
        for i in range(len(outputs)):
            # Concatenate the ith elements of outputs and labels
            concatenated = np.concatenate((outputs[i], ohe_labels[i]))
            concatenated_list.append(concatenated)

        concatenated = np.array(concatenated_list)

        return [inputs, concatenated]

    def read_data_conventional(self):

        print("----------PREPROCESSIING DATASET------------")

        inputs = np.load(conventional_input)
        outputs = np.load(conventional_output)

        # Reshape the labels to a 2D array
        outputs, labels = outputs[:, :-1], outputs[:, -1]

        labels_reshaped = list(labels)


        ohe_labels = self.custom_one_hot_encode(labels_reshaped)

        # Initialize an empty list to store concatenated elements
        concatenated_list = []

        # Loop through each element (i) of outputs and labels
        for i in range(len(outputs)):
            # Concatenate the ith elements of outputs and labels
            concatenated = np.concatenate((outputs[i], ohe_labels[i]))
            concatenated_list.append(concatenated)

        concatenated = np.array(concatenated_list)

        return [inputs, concatenated]

    def train_test_split_dataset(self, type):

        # TODO Change for dataset here
        if type == "PMU_caseA":
            inputs, outputs = self.read_data_PMU_caseA()
            X_train_file = X_train_PMU_caseA
            y_train_file = y_train_PMU_caseA
            X_val_file   = X_val_PMU_caseA
            y_val_file   = y_val_PMU_caseA
            X_test_file  = X_test_PMU_caseA
            y_test_file  = y_test_PMU_caseA
        elif type == "PMU_caseB":
            inputs, outputs = self.read_data_PMU_caseB()
            X_train_file = X_train_PMU_caseB
            y_train_file = y_train_PMU_caseB
            X_val_file = X_val_PMU_caseB
            y_val_file = y_val_PMU_caseB
            X_test_file = X_test_PMU_caseB
            y_test_file = y_test_PMU_caseB
        elif type == "conventional":
            inputs, outputs = self.read_data_conventional()
            X_train_file = X_train_conventional
            y_train_file = y_train_conventional
            X_val_file = X_val_conventional
            y_val_file = y_val_conventional
            X_test_file = X_test_conventional
            y_test_file = y_test_conventional
        else:
            print("Enter meter type")
            sys.exit(0)

        if (dataset == "MESOGEIA") and (meterType=="PMU_caseB"):
            # First split: train+validation and test

            # Print shape of inputs-outputs
            print(inputs.shape, outputs.shape)

            topology_offset = 7787

            train_T1_start, train_T1_end = 0, 7307
            train_T2_start, train_T2_end = topology_offset + 0, topology_offset + 0 + 7307
            print(train_T1_start, train_T1_end, train_T2_start, train_T2_end)

            test_T1_start, test_T1_end = 7307, 7307+96
            test_T2_start, test_T2_end = topology_offset + 7307+96, topology_offset + 7307+96+96
            print(test_T1_start, test_T1_end, test_T2_start, test_T2_end)

            val_T1_start, val_T1_end = 7307+96+96, topology_offset
            val_T2_start, val_T2_end = topology_offset + 7307+96+96, topology_offset + topology_offset
            print(val_T1_start, val_T1_end, val_T2_start, val_T2_end)

            print(inputs.shape, outputs.shape)


            X_train = np.concatenate([inputs[train_T1_start:train_T1_end, :], inputs[train_T2_start:train_T2_end, :]])
            print(X_train.shape)
            y_train = np.concatenate([outputs[train_T1_start:train_T1_end, :], outputs[train_T2_start:train_T2_end, :]])

            X_test  = np.concatenate([inputs[test_T1_start:test_T1_end, :], inputs[test_T2_start:test_T2_end, :]])
            y_test  = np.concatenate([outputs[test_T1_start:test_T1_end, :], outputs[test_T2_start:test_T2_end, :]])

            X_val   = np.concatenate([inputs[val_T1_start:val_T1_end, :], inputs[val_T2_start:val_T2_end, :]])
            y_val   = np.concatenate([outputs[val_T1_start:val_T1_end, :], outputs[val_T2_start:val_T2_end, :]])

        else:
            # First split: train+validation and test
            X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.15, random_state=42)

            # Second split: train and validation
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)  # 0.25 x 0.8 = 0.2

        if type == "PMU_caseA":

            meter = 75

            X_old = X_test.copy()

            # TODO - Outlier injection into the datasets X_train, X_test
            out_inj_X_train = OutlierInjector(X_train)
            X_train_outlier_arr, X_train_outlier_index_dict = out_inj_X_train.inject_outliers_PMUcaseA([meter],
                                                                                                       prob_m=0.01,
                                                                                                       prob_a=0.01)
            out_inj_X_test = OutlierInjector(X_test)
            X_test_outlier, X_test_outlier_index_dict = out_inj_X_test.inject_outliers_PMUcaseA([meter],
                                                                                                prob_m=0.01,
                                                                                                prob_a=0.01)
            print("X_test shape: ", X_test.shape)
            print("y_test shape: ", y_test.shape)
            print("X_test outlier shape: ", X_test_outlier.shape)
            # TODO - Isolation Forest Outlier Detection - Gets outlier X_train dataset as input
            iso_detector = IF_OutlierDetection(meter)
            iso_detector.train_isolation_forest_PMU_caseA(X_train_outlier_arr)
            per_edge_outliers = iso_detector.predict_outliers_PMU_caseA(X_test_outlier, X_test_outlier_index_dict)

            # TODO - KNN imputer to impute values
            knn_imp = KNNImputerMeasurements(meter)
            knn_imp.train_KNNImputers_PMU_caseA(X_train_outlier_arr, X_train_outlier_index_dict)
            X_test_imputed = knn_imp.impute_values_PMU_caseA(X_test_outlier, X_test_outlier_index_dict,
                                                             compare_np=X_old)

            # TODO - From X_imputed and t_test, we need to remove the indices of all PMU measurements wrong
            X_test_imputed = X_test_imputed[~np.isin(np.arange(len(X_test_imputed)), per_edge_outliers["ALL"])]
            y_test_imputed = y_test[~np.isin(np.arange(len(y_test)), per_edge_outliers["ALL"])]
            print("X_test imputed shape: ", X_test_imputed.shape)
            print("y_test imputed shape: ", y_test_imputed.shape)
            # TODO Here add y_test ALL_outlier indices exclusion

            np.save("datasets/95UKGD_PMU_caseA_X_train_unscaled.npy", X_train)
            np.save("datasets/95UKGD_PMU_caseA_y_train_unscaled.npy", y_train)
            np.save("datasets/95UKGD_PMU_caseA_X_val_unscaled.npy", X_val)
            np.save("datasets/95UKGD_PMU_caseA_y_val_unscaled.npy", y_val)
            np.save("datasets/95UKGD_PMU_caseA_X_test_unscaled.npy", X_test)
            np.save("datasets/95UKGD_PMU_caseA_y_test_unscaled.npy", y_test)


            #TODO Divide below code into 2 separate
            scaler          = StandardScaler()
            X_train         = scaler.fit_transform(X_train)
            X_val           = scaler.transform(X_val)
            X_test          = scaler.transform(X_test)
            X_test_outlier  = scaler.transform(X_test_outlier)
            X_test_imputed  = scaler.transform(X_test_imputed)

            np.save(X_train_file, X_train)
            np.save(y_train_file, y_train)
            np.save(X_val_file, X_val)
            np.save(y_val_file, y_val)
            np.save(X_test_file, X_test)
            np.save(y_test_file, y_test)
            np.save(X_test_PMU_caseA_outliers, X_test_outlier)
            np.save(X_test_PMU_caseA_imputed, X_test_imputed)
            np.save(y_test_PMU_caseA_imputed, y_test_imputed)

            print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test_outlier.shape, X_test_imputed.shape, y_test_imputed.shape)


        else:
            scaler   = StandardScaler()
            X_old    = X_train.copy()
            X_train  = scaler.fit_transform(X_train)
            X_val    = scaler.transform(X_val)
            X_test   = scaler.transform(X_test)
            joblib.dump(scaler, 'scalers/MESOGEIA_NN_PMUB_METERS_StandardScaler_all_features.pkl')

            if dataset == "MESOGEIA":
                used_feature_indices = sorted([128, 258, 259, 255, 389, 390, 386, 520, 521, 517, 124, 127])
                print(used_feature_indices)

                X_dep = X_old[:, used_feature_indices]
                dep_scaler = StandardScaler()
                X_dep = dep_scaler.fit_transform(X_dep)

                #TODO Save scaler
                joblib.dump(dep_scaler, 'scalers/MESOGEIA_NN_PMUB_METERS_StandardScaler.pkl')


            np.save(X_train_file, X_train)
            np.save(y_train_file, y_train)
            np.save(X_val_file, X_val)
            np.save(y_val_file, y_val)
            np.save(X_test_file, X_test)
            np.save(y_test_file, y_test)

    def preprocess_data(self, type):

        if type == "PMU_caseA":

            X_train = np.load(X_train_PMU_caseA)
            y_train = np.load(y_train_PMU_caseA)
            X_val = np.load(X_val_PMU_caseA)
            y_val = np.load(y_val_PMU_caseA)
            X_test = np.load(X_test_PMU_caseA)
            y_test = np.load(y_test_PMU_caseA)

            #TODO Read outlier datasets
            X_test_outliers = np.load(X_test_PMU_caseA_outliers)
            X_test_imputed  = np.load(X_test_PMU_caseA_imputed)
            y_test_imputed  = np.load(y_test_PMU_caseA_imputed)

        elif type == "PMU_caseB":
            X_train = np.load(X_train_PMU_caseB)
            y_train = np.load(y_train_PMU_caseB)
            X_val = np.load(X_val_PMU_caseB)
            y_val = np.load(y_val_PMU_caseB)
            X_test = np.load(X_test_PMU_caseB)
            y_test = np.load(y_test_PMU_caseB)

        elif type == "conventional":
            X_train = np.load(X_train_conventional)
            y_train = np.load(y_train_conventional)
            X_val = np.load(X_val_conventional)
            y_val = np.load(y_val_conventional)
            X_test = np.load(X_test_conventional)
            y_test = np.load(y_test_conventional)

        else:
            print("Please enter known meter Type")
            sys.exit(0)

        #TODO Divide SE/TI output
        y_train_outputs = y_train[:, :2 * NUM_NODES]
        y_train_labels = y_train[:, 2 * NUM_NODES:]

        y_val_outputs = y_val[:, :2 * NUM_NODES]
        y_val_labels = y_val[:, 2 * NUM_NODES:]

        y_test_outputs = y_test[:, :2 * NUM_NODES]
        y_test_labels = y_test[:, 2 * NUM_NODES:]

        if type == "PMU_caseA":
            y_test_imputed_outputs = y_test_imputed[:, :2 * NUM_NODES]
            y_test_imputed_labels = y_test_imputed[:, 2 * NUM_NODES:]


        if type == "PMU_caseA":
            return X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs,\
                y_test_labels, X_test_outliers, X_test_imputed, y_test_imputed_outputs, y_test_imputed_labels
        else:
            return X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, \
                y_test_labels, None, None, None, None


    def preprocess_meter_type(self, type):
        if type == "PMU_caseA":
            # TODO Case A - Store then read for each measurement Vm, Va, Im, Ia
            #self.store_data_PMU_caseA()
            X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, \
                y_test_labels, X_test_outliers, X_test_imputed, y_test_imputed_outputs, y_test_imputed_labels = self.preprocess_data("PMU_caseA")
        elif type == "PMU_caseB":
            # TODO Case B - Store then read for each measurement Vm, Va, Iinjm, Iinja
            #self.store_data_PMU_caseB()
            X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, y_test_labels, a, b, c, d = self.preprocess_data("PMU_caseB")
        elif type == "conventional":
            # TODO Case B - Store then read for each measurement Vm, Pinj, Qinj
            #self.store_data_conventional()
            X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, y_test_labels = self.preprocess_data("conventional")
        else:
            print("Please enter known meter type")
            sys.exit(0)

        if type == "PMU_caseA":
            return X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, \
                y_test_labels, X_test_outliers, X_test_imputed, y_test_imputed_outputs, y_test_imputed_labels
        else:
            return X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, \
                y_test_labels, a, b, c, d

class BuildModel:

    def __init__(self, NN_class):

        self.model=NN_class

    def build_simple_nn(self, input_dim):
        # Input layer
        inputs = layers.Input(shape=(input_dim,))

        # First hidden layer with 64 neurons and ReLU activation
       # x = layers.Dense(128, activation='relu')(inputs)

        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(16, activation='relu')(x)

        # First hidden layer with 64 neurons and ReLU activation
        #x = layers.Dense(16, activation='relu')(x)

        # Second hidden layer with 32 neurons and ReLU activation
        #x = layers.Dense(16, activation='relu')(x)

        # Output layer (num_classes corresponds to the number of output classes)
        outputs = layers.Dense(NUM_TOPOLOGIES, activation='softmax')(x)

        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile the model with categorical cross-entropy loss and Adam optimizer
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_model(self, input_dim):

        if self.model=="NN":
            return self.build_simple_nn(input_dim)

class TrainModel:

    def __init__(self, model, X_train, y_train, X_val, y_val):

        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, = X_train, y_train, X_val, y_val

    def train_model(self):

        early_stopping = EarlyStopping(
            monitor='val_loss',  # You can use 'val_accuracy' or any other metric you are monitoring
            patience=20,  # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore model weights from the epoch with the best metric
        )

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model and save the history
        history = self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=16, validation_data=(self.X_val, self.y_val), callbacks=[early_stopping])

        # Plot training & validation accuracy and loss values
        plt.figure(figsize=(14, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
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
        plt.savefig('TI_train_plot.png')

        return self.model

class PreProcFS:

    def __init__(self, meterType, Preproc_model, submethod, X_train, y_train):

        self.meterType          = meterType
        self.Preproc_model      = Preproc_model
        self.submethod  = submethod
        self.X_train    = X_train
        self.y_train    = y_train

    def execute_rfe_rf_PMU_caseA(self):

        remaining_branches = [i for i in range(NUM_BRANCHES)]
        if dataset == "MESOGEIA":
            remaining_branches = BRANCH_PICK_LIST

        EXISTING_METER_BRANCHES = []


        #TODO Exclude existing branches
        feature_group_dict = {b_i: [  0*NUM_NODES + branch_data[b_i]["sending_node"],
                                      1*NUM_NODES + branch_data[b_i]["sending_node"],
                                      2*NUM_NODES + b_i,
                                      2*NUM_NODES+NUM_BRANCHES + b_i] for b_i in remaining_branches}

        remaining_branches = [i for i in remaining_branches if i not in EXISTING_METER_BRANCHES]

        num_features = 4
        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)

        #TODO - Remember for original meters
        used_branches = []
        if EXISTING_METER_BRANCHES!=[]:
            # TODO Train RF initially with EXISTING NODES to find the residual
            # print(self.X_train[0], self.y_train[0])
            y_labels = np.argmax(self.y_train, axis=1).reshape(-1)
            existing_meter_indices = []
            for branch in EXISTING_METER_BRANCHES:
                existing_meter_indices.extend((feature_group_dict[branch]))
            X_train = X_train_init[:, existing_meter_indices]
            print(X_train[0], X_train.shape)
            rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_labels)
            y_pred = rf.predict(X_train)
            misclassified = (y_pred != y_labels)

        while len(remaining_branches) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_branches:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test  = X_test_init[:, used_indices]

            #print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            test_accuracy = accuracy_score(pred, y_test)
            print("Test accuracy of RF: ", test_accuracy)

            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_branches))]
            #print("Added importances per group: ", group_importance_list)
            min_importance_branch = group_importance_list.index(min(group_importance_list))
            #print("Minimum importance for worst group: ", min(group_importance_list))
            #print("Chose: ", min_importance_branch)
            #print(remaining_branches, min_importance_branch)
            real_branch_index = remaining_branches.pop(min_importance_branch)
            print("Min importance branch index in remaining list: ", min_importance_branch, "Min real index: ", real_branch_index)
            used_branches.append(real_branch_index)

        used_branches.reverse()

        #TODO Place first existing branches
        used_branches = EXISTING_METER_BRANCHES + used_branches

        print("TI Feature Order: ", used_branches)

        # return sorted_indices
        return used_branches

    def execute_rfe_rf_PMU_caseB(self):

        remaining_nodes = [i for i in range(NUM_NODES)]
        if dataset == "MESOGEIA":
            remaining_nodes = NODE_PICK_LIST

        #TODO Exclude existing nodes
        #remaining_nodes = [i for i in remaining_nodes if i not in EXISTING_METER_NODES]

        feature_group_dict = {n_i: [  0*NUM_NODES + n_i,
                                      1*NUM_NODES + n_i,
                                      2*NUM_NODES + n_i,
                                      3*NUM_NODES + n_i] for n_i in remaining_nodes}
        num_features = 4
        used_nodes = [] #EXISTING_METER_NODES

        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)

        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test = X_test_init[:, used_indices]

            # print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            test_accuracy = accuracy_score(pred, y_test)
            print("Test accuracy of RF: ", test_accuracy)
            importances = rf.feature_importances_
            print("Importances: ", importances)
            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_nodes))]
            print("Added importances per group: ", group_importance_list)
            min_importance_node = group_importance_list.index(min(group_importance_list))
            print("Minimum importance for worst group: ", min(group_importance_list))
            print("Chose: ", min_importance_node)
            print("Remaining nodes: ", remaining_nodes)
            print("Minimum importance index: ", min_importance_node)
            real_node_index = remaining_nodes.pop(min_importance_node)
            print("Min importance branch index in remaining list: ", min_importance_node, "Min real index: ", real_node_index)
            used_nodes.append(real_node_index)
            print(used_nodes)

        used_nodes.reverse()

        #TODO Place first existing branches
        #used_nodes = EXISTING_METER_BRANCHES + used_nodes

        print("TI Feature Order: ", used_nodes)

        # return sorted_indices
        return used_nodes

    def execute_rfe_rf_conventional(self):

        remaining_nodes = [i for i in range(NUM_NODES)]
        if dataset == "MESOGEIA":
            remaining_nodes = NODE_PICK_LIST
        #TODO Exclude existing nodes
        #remaining_nodes = [i for i in remaining_nodes if i not in EXISTING_METER_NODES]

        feature_group_dict = {n_i: [  0*NUM_NODES + n_i,
                                      1*NUM_NODES + n_i,
                                      2*NUM_NODES + n_i] for n_i in remaining_nodes}
        num_features = 3

        EXISTING_METER_NODES = []
        used_nodes = EXISTING_METER_NODES

        X_train_init, X_test_init, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=42)


        while len(remaining_nodes) > 0:

            used_indices = []

            # Select Features per group to be used on RF training from remaining branches
            # Select Features per group to be used on RF training from remaining branches
            for b_i in remaining_nodes:
                used_indices.extend(feature_group_dict[b_i])

            X_train = X_train_init[:, used_indices]
            X_test  = X_test_init[:, used_indices]

            #print("Used indices: ", used_indices)

            # Train Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=TREE_DEPTH, random_state=42)
            rf.fit(X_train, y_train)

            pred = rf.predict(X_test)
            test_accuracy = accuracy_score(pred, y_test)
            print("Test accuracy of RF: ", test_accuracy)

            importances = rf.feature_importances_

            #TODO How to return from list of new indices back to the original ones
            # Remove from the original list the element of the worst index
            # Get back as remaining_features[j]
            group_importance_list = [sum(importances[num_features*i:(i+1)*num_features]) for i in range(len(remaining_nodes))]
            print("Added importances per group: ", group_importance_list)
            min_importance_node = group_importance_list.index(min(group_importance_list))
            print("Minimum importance for worst group: ", min(group_importance_list))
            print("Chose: ", min_importance_node)
            print(remaining_nodes, min_importance_node)
            real_node_index = remaining_nodes.pop(min_importance_node)
            print("Min importance branch index in remaining list: ", min_importance_node, "Min real index: ", remaining_nodes)
            used_nodes.append(real_node_index)
            print(used_nodes)

        used_nodes.reverse()

        #TODO Place first existing branches
        #used_nodes = EXISTING_METER_BRANCHES + used_nodes

        print("TI Feature Order: ", used_nodes)

        # return sorted_indices
        return used_nodes

    def execute_pca_magnitudes_and_angles(self):

        # Initialize PCA
        pca = PCA(n_components=136)  # Keep all components for now
        X_pca = pca.fit_transform(self.X_train)

        # Explained variance ratio for each component
        explained_variances = pca.explained_variance_ratio_

        loadings = pca.components_

        used_branches = []
        all_indices = [i for i in range(35)]
        for _ in range(35):
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
                    for i in [sending_node_index, NUM_NODES + sending_node_index, 2*NUM_NODES + branch_index, NUM_NODES+NUM_BRANCHES + branch_index]:
                        tmp_list.append(i)
                else:
                    for i in [2*NUM_NODES + branch_index, 2*NUM_NODES + NUM_BRANCHES + branch_index]:
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
        if self.Preproc_model=="RF":
            if self.submethod == "rfe":
                if self.meterType == "PMU_caseA":
                    return self.execute_rfe_rf_PMU_caseA()
                elif self.meterType == "PMU_caseB":
                    return self.execute_rfe_rf_PMU_caseB()
                if self.meterType == "conventional":
                    return self.execute_rfe_rf_conventional()

            #elif self.submethod == "sum":
            #    return self.execute_rf_model_magnitudes_and_angles_sum()
            #elif self.submethod == "max":
            #    return self.execute_rf_model_magnitude_angles_max()

            else:
                print("Invalid model - submethod combination")


        #elif self.Preproc_model=="GNN":
        #    if self.submethod == "simple":
        #        return self.execute_gnn_magnitudes_angles_fs()


        elif self.Preproc_model=="PCA":
            if self.submethod == "simple":
                return self.execute_pca_magnitudes_and_angles()

class TIPredictorTrainProcess:

    def __init__(self, meterType, threshold, model, X_train, y_train, X_val, y_val, X_test, y_test, FS="RF",
                 method="sum", iterative_fs=False, X_test_outliers=None, X_test_imputed=None, y_test_imputed=None):

        self.meterType = meterType
        self.threshold = threshold
        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = X_train, y_train, X_val, y_val, X_test, y_test
        self.X_test_outliers, self.X_test_imputed, self.y_test_imputed = X_test_outliers, X_test_imputed, y_test_imputed
        if self.meterType == "PMU_caseA":
            self.num_features = 2
        elif self.meterType == "PMU_caseB":
            self.num_features = 4
        elif self.meterType == "conventional":
            self.num_features = 3
        else:
            self.num_features = None
        self.FS = FS
        self.method = method
        self.iterative_fs = iterative_fs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_simple_NN(self, model, X_test, y_test):

        correct_predictions = 0
        total_samples = 0

        # Forward pass
        out_test = self.model(X_test)

        print(out_test.shape, self.y_test.shape)

        # Get predicted class by selecting the index of the maximum logit for each sample
        preds = [np.array(i).argmax() for i in out_test]
        true_labels = [np.array(i).argmax() for i in y_test]



        # Calculate the number of correct predictions
        correct_predictions = sum([1 if (preds[i] == true_labels[i]) else 0 for i in range(out_test.shape[0])])
        total_samples = len(preds)

        return correct_predictions / total_samples

    def execute_NN(self):

        if not self.iterative_fs:
            print(self.X_train.shape)
            FS = PreProcFS(self.meterType, self.FS, self.method, self.X_train, self.y_train)
            #features = FS.execute()
            if self.meterType == "PMU_caseA":
                features = UKGD95_PMU_caseA_TI_features
                features = features
                print("TI Feature Selection Order - Branches: ", features)
            elif self.meterType == "PMU_caseB":
                features = MESOGEIA_PMU_caseB_TI_features
                features = features
                print("TI Feature Selection Order - Nodes: ", features)
            elif self.meterType == "conventional":
                features = IEEE33_conventional_TI_features
                features = features
                print("TI Feature Selection Order - Nodes: ", features)

            #TODO For every Currenct branch input feature add the magnitude and its angle
            # If the magnitude is at index X, then angle is at index X+35
            used_feature_indices = []
            acc_meters = []
            for i in features:
                acc_meters.append(i)
                if self.meterType == "PMU_caseA":
                    used_feature_indices.append(i)
                    print("Chose feature - branch: ", i, "Total feature indices: ", used_feature_indices)
                    node_indices = [branch_data[i]["sending_node"] for i in used_feature_indices] + [NUM_NODES + branch_data[i]["sending_node"] for i in used_feature_indices]
                    current_indices = [i+2*NUM_NODES for i in used_feature_indices]+[2*NUM_NODES+i+NUM_BRANCHES for i in used_feature_indices]
                    all_indices = node_indices + current_indices
                elif self.meterType == "PMU_caseB":
                    used_feature_indices.append(i)
                    #TODO Insert manually
                    #used_feature_indices = [127, 128, 112, 117] Case B
                    print("Chose feature - node: ", i, "Total feature indices: ", used_feature_indices)
                    node_indices = [i for i in used_feature_indices] + \
                                   [NUM_NODES + i for i in used_feature_indices] + \
                                   [2*NUM_NODES + i for i in used_feature_indices] + \
                                   [3*NUM_NODES + i for i in used_feature_indices]
                    all_indices = sorted(node_indices)
                elif self.meterType == "conventional":
                    used_feature_indices.append(i)

                    #TODO - Static TI
                    #used_feature_indices = [27, 11, 7, 28, 13, 21, 24, 12, 29, 6, 9, 8, 26, 30, 17, 20, 16, 32, 14, 31, 25]

                    print("Chose feature - node: ", i, "Total feature indices: ", used_feature_indices)
                    node_indices = [i for i in used_feature_indices] + \
                                   [NUM_NODES + i for i in used_feature_indices] + \
                                   [2*NUM_NODES + i for i in used_feature_indices]

                    all_indices = node_indices
                X_train = self.X_train[:, all_indices]
                X_val   = self.X_val[:, all_indices]
                X_test  = self.X_test[:, all_indices]

                #TODO Simple NN with keras
                ML_model = "NN"
                buildModel = BuildModel(ML_model)
                input_dim = len(all_indices)
                print(all_indices)
                self.model = buildModel.build_model(input_dim)
                print(X_train.shape, self.y_train.shape)
                trainModel = TrainModel(self.model, X_train, self.y_train, X_val, self.y_val)
                trainModel.train_model()
                # Evaluate the model on the test data
                test_loss, test_accuracy = self.model.evaluate(X_test, self.y_test, verbose=0)
                test_accuracy = self.evaluate_simple_NN(self.model, X_test, self.y_test)

                print("Meters: ", acc_meters, " - Accuracy: ", test_accuracy)

                NNdimension = len(used_feature_indices)

                #ML_model   = "NN"
                #if self.meterType == "PMU_caseA":
                #    num_node_features = 2
                #    ANN = TI_SimpleNNEdges(NNdimension,num_node_features,NUM_TOPOLOGIES, branch_num=NNdimension, branch_feature_num=2).to(self.device)
                #else:
                #    ANN = TI_SimpleNNEdges(NNdimension, self.num_features, NUM_TOPOLOGIES, branch_num=None, branch_feature_num=None).to(self.device)
                #trainModel = TrainNN_TI(ANN,X_train,self.y_train,X_val,self.y_val, X_test, self.y_test, NUM_TOPOLOGIES)
                #print("X_train shape: ", X_train.shape)
                #test_accuracy = trainModel.evaluate()

                #print(used_feature_indices, test_accuracy)
                #with open("results.txt", "a") as wf:
                #    wf.write("USed indices (i+1 for proper index): "+str([i+1 for i in used_feature_indices])+", Accuracy: "+str(test_accuracy)+"\n")
                #    wf.close()

                filename = "DeployedModels/" + f"MESOGEIA_NN_METERS_{str('_'.join([str(x) for x in acc_meters]))}_TI.h5"
                self.model.save(filename)

                #with open(filename, "a") as wf:
                #    wf.write("Used indices (i+1 for proper index): "+str([i+1 for i in used_feature_indices])+", Accuracy: "+str(test_accuracy)+"\n")
                #    wf.close()

                if test_accuracy >= self.threshold: break


            return used_feature_indices

    def execute_GNN(self):

        if not self.iterative_fs:
            FS = PreProcFS(self.meterType, self.FS, self.method, self.X_train, self.y_train)
            #Ib_features = FS.execute()
            #Ib_features = GLOBAL_BRANCH_LIST
            Ib_features = []
            if self.meterType == "PMU_caseA":
                Ib_features = UKGD95_PMU_caseA_TI_features
                print("TI Feature Selection Order - Branches: ", Ib_features)
            elif self.meterType == "PMU_caseB":
                Ib_features = UKGD95_PMU_caseB_TI_features #IEEE33_PMU_caseB_TI_features
                #Ib_features = Ib_features
                print("TI Feature Selection Order - Nodes: ", Ib_features)
            elif self.meterType == "conventional":
                Ib_features = UKGD95_conventional_TI_features #IEEE33_conventional_TI_features
                print("TI Feature Selection Order - Nodes: ", Ib_features)
            print("TI Feature Selection Order: ", Ib_features)

            #TODO For every Currenct branch input feature add the magnitude and its angle
            # If the magnitude is at index X, then angle is at index X+35
            used_feature_indices = []
            for i in Ib_features:
                #print("Chose Ibranch: ", i)
                used_feature_indices.append(i)
                #TODO FOr a certain meter only
                #used_feature_indices = [27, 11, 7, 28, 13, 21, 24, 12, 29, 6, 9, 8, 26, 30, 17, 20, 16, 32, 14, 31, 25]

                X_train_TI = self.X_train
                y_train_labels = self.y_train

                X_val_TI = self.X_val
                y_val_labels = self.y_val

                X_test_TI = self.X_test
                y_test_labels = self.y_test

                X_test_outliers = self.X_test_outliers
                X_test_imputed  = self.X_test_imputed
                y_test_imputed  = self.y_test_imputed


                print("EDGE Indices: ", used_feature_indices)
                GTIP = GraphTIPreprocess_IV(self.meterType, used_feature_indices, X_train_TI, y_train_labels, X_val_TI,
                                            y_val_labels, X_test_TI, y_test_labels, X_test_outliers, X_test_imputed, y_test_imputed)

                if self.meterType == "PMU_caseA":
                    edges, train_loader, validation_loader, test_loader, test_outliers_loader, \
                        test_imputed_loader = GTIP.generate_dataset_TI_GNN_PMU_caseA()
                    print(train_loader)
                    Train_GNN_TI = TrainGNN_TI(self.meterType, self.model, used_feature_indices, train_loader,
                                               validation_loader, test_loader, test_outliers_loader, test_imputed_loader)
                    acc = Train_GNN_TI.evaluate()
                elif self.meterType == "PMU_caseB":
                    edges, train_loader, validation_loader, test_loader = GTIP.generate_dataset_TI_GNN_PMU_caseB()
                    Train_GNN_TI = TrainGNN_TI(self.meterType, self.model, used_feature_indices, train_loader,
                                               validation_loader, test_loader, None, None)
                    acc = Train_GNN_TI.evaluate()
                elif self.meterType == "conventional":
                    edges, train_loader, validation_loader, test_loader = GTIP.generate_dataset_TI_GNN_conventional()
                    Train_GNN_TI = TrainGNN_TI(self.meterType, self.model, used_feature_indices, train_loader,
                                               validation_loader, test_loader, None, None)
                    acc = Train_GNN_TI.evaluate()

                filename = "results/" + "TI___MODEL___GNN_simple___" + "PREPROCESSING_" + str(FS.Preproc_model) + "___SUBMETHOD___" + str(FS.submethod) + "_results.txt"
                print(filename)

                with open(filename, "a") as wf:
                    wf.write("Used indices (i+1 for proper index): " + str(
                        [i + 1 for i in used_feature_indices]) + ", Accuracy: " + str(acc) + "\n")
                    wf.close()

                if acc >= self.threshold: break

            return used_feature_indices

    def execute(self):

        if self.model == "NN":
            # Return branch index list (i)
            return self.execute_NN()
        elif self.model == "GNN":
            # Return branch index list (i)
            return self.execute_GNN()

class GraphTIPreprocess_IV:

    def __init__(self, meterType, selected_edges, X_train, y_train, X_val, y_val, X_test, y_test,
                 X_test_outliers=None, X_test_imputed=None, y_test_imputed=None):

        self.meterType             = meterType
        self.branch_data           = branch_data
        self.edge_index            = None
        self.selected_edges        = selected_edges
        self.X_train               = X_train
        self.y_train               = y_train
        self.X_val                 = X_val
        self.y_val                 = y_val
        self.X_test                = X_test
        self.y_test                = y_test
        self.X_test_outliers       = X_test_outliers
        self.X_test_imputed        = X_test_imputed
        self.y_test_imputed        = y_test_imputed

    def define_graph(self):
        # Complete edge_index for all branches (35 edges in total)
        edges = [(v['sending_node'], v['receiving_node']) for v in branch_data.values()]
        #edges = [edges[i] for i in self.selected_edges]
        #print("Selected edges: ", selected_edges)
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Ensure correct shape [2, num_edges]
        return self.edge_index

    # Preprocessing to select features for the specific edges used
    def preprocess_data(self, NP_train, selected_edge_indices, num_edges=NUM_BRANCHES, num_features_per_edge=2, num_nodes=NUM_NODES, num_features_per_node=2):

        # Edges start at index Vm: 0 - 32, Va: 33 - 65, Im: 66 - 100, Ia: 101 - 135

        branch_index_offset = 2*NUM_NODES
        node_index_offset   = 0

        edge_features   = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))
        edge_mask       = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))

        node_features   = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))
        node_mask       = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))

        # Only the features for selected edges will be non-zero
        for idx, edge_idx in enumerate(selected_edge_indices):
            mag_feature = NP_train[:, branch_index_offset + edge_idx]
            angle_feature = NP_train[:, branch_index_offset + edge_idx + 35]
            edge_features[:, edge_idx * 2] = mag_feature
            edge_features[:, edge_idx * 2 + 1] = angle_feature
            edge_mask[:, edge_idx * 2] = 1
            edge_mask[:, edge_idx * 2 + 1] = 1

            node_idx = branch_data[edge_idx]["sending_node"]
            mag_feature = NP_train[:, node_index_offset + node_idx]
            angle_feature = NP_train[:, node_index_offset + node_idx + 33]
            node_features[:, node_idx * 2] = mag_feature
            node_features[:, node_idx * 2 + 1] = angle_feature
            node_mask[:, node_idx * 2] = 1
            node_mask[:, node_idx * 2 + 1] = 1

        return edge_features, edge_mask, node_features, node_mask

    def preprocess_data_PMU_caseA(self, NP_train, selected_edge_indices, num_edges, num_features_per_edge=2, num_nodes=NUM_NODES, num_features_per_node=2):

        # Edges start at index Vm: 0 - 32, Va: 33 - 65, Im: 66 - 100, Ia: 101 - 135

        branch_index_offset = 2*NUM_NODES
        node_index_offset   = 0

        edge_features   = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))
        edge_mask       = np.zeros((NP_train.shape[0], num_edges * num_features_per_edge))

        node_features   = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))
        node_mask       = np.zeros((NP_train.shape[0], num_nodes * num_features_per_node))

        # Only the features for selected edges will be non-zero
        for idx, edge_idx in enumerate(selected_edge_indices):
            mag_feature = NP_train[:, branch_index_offset + edge_idx]
            angle_feature = NP_train[:, branch_index_offset + edge_idx + NUM_BRANCHES]
            edge_features[:, edge_idx * num_features_per_edge] = mag_feature
            edge_features[:, edge_idx * num_features_per_edge + 1] = angle_feature
            edge_mask[:, edge_idx * num_features_per_edge] = 1
            edge_mask[:, edge_idx * num_features_per_edge + 1] = 1

            node_idx = branch_data[edge_idx]["sending_node"]
            mag_feature = NP_train[:, node_index_offset + node_idx]
            angle_feature = NP_train[:, node_index_offset + node_idx + NUM_NODES]
            node_features[:, node_idx * num_features_per_node] = mag_feature
            node_features[:, node_idx * num_features_per_node + 1] = angle_feature
            node_mask[:, node_idx * num_features_per_node] = 1
            node_mask[:, node_idx * num_features_per_node + 1] = 1

        return edge_features, edge_mask, node_features, node_mask

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

    def generate_dataset_TI_GNN_PMU_caseA(self):

        num_active_branches   = len(self.selected_edges)
        num_edges             = NUM_BRANCHES
        num_nodes             = NUM_NODES
        num_features_node     = 2
        num_features_edge     = 2

        #self.selected_edges = [6, 32, 9]
        print(self.X_test_outliers.shape)
        print(self.X_test_imputed.shape)
        print(self.y_test_imputed.shape)

        train_edge_data, train_edge_mask, train_node_data, train_node_mask = self.preprocess_data_PMU_caseA(self.X_train, self.selected_edges, num_edges, num_features_edge, num_nodes, num_features_node)
        val_edge_data, val_edge_mask, val_node_data, val_node_mask   = self.preprocess_data_PMU_caseA(self.X_val, self.selected_edges, num_edges, num_features_edge, num_nodes, num_features_node)
        test_edge_data, test_edge_mask, test_node_data, test_node_mask  = self.preprocess_data_PMU_caseA(self.X_test, self.selected_edges, num_edges, num_features_edge, num_nodes, num_features_node)
        test_edge_outlier_data, test_edge_outlier_mask, test_node_outlier_data, test_node_outlier_mask  = self.preprocess_data_PMU_caseA(self.X_test_outliers, self.selected_edges, num_edges, num_features_edge, num_nodes, num_features_node)
        test_edge_imputed_data, test_edge_imputed_mask, test_node_imputed_data, test_node_imputed_mask  = self.preprocess_data_PMU_caseA(self.X_test_imputed, self.selected_edges, num_edges, num_features_edge, num_nodes, num_features_node)


        #print("X train shape: ", self.X_train.shape)
        #print("X val shape: ", self.X_val.shape)
        #print("X test shape: ", self.X_test.shape)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):
            tmp_edge_attr = torch.tensor(train_edge_data[i].reshape(-1, num_features_edge), dtype=torch.float)
            tmp_edge_mask = torch.tensor(train_edge_mask[i].reshape(-1, num_features_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, num_features_node), dtype=torch.float)
            tmp_node_mask = torch.tensor(train_node_mask[i].reshape(-1, num_features_node), dtype=torch.float)


            label = torch.tensor(self.y_train[i, :], dtype=torch.float)
            #print(edge_index, edge_attr, mask, label)
            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):
            tmp_edge_attr = torch.tensor(val_edge_data[i].reshape(-1, num_features_edge), dtype=torch.float)
            tmp_edge_mask = torch.tensor(val_edge_mask[i].reshape(-1, num_features_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, num_features_node), dtype=torch.float)
            tmp_node_mask = torch.tensor(val_node_mask[i].reshape(-1, num_features_node), dtype=torch.float)

            label = torch.tensor(self.y_val[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):
            tmp_edge_attr = torch.tensor(test_edge_data[i].reshape(-1, num_features_edge), dtype=torch.float)
            tmp_edge_mask = torch.tensor(test_edge_mask[i].reshape(-1, num_features_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, num_features_node), dtype=torch.float)
            tmp_node_mask = torch.tensor(test_node_mask[i].reshape(-1, num_features_node), dtype=torch.float)

            label = torch.tensor(self.y_test[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_outlier_data = []
        for i in range(self.X_test_outliers.shape[0]):
            tmp_edge_attr = torch.tensor(test_edge_outlier_data[i].reshape(-1, num_features_edge), dtype=torch.float)
            tmp_edge_mask = torch.tensor(test_edge_outlier_mask[i].reshape(-1, num_features_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(test_node_outlier_data[i].reshape(-1, num_features_node), dtype=torch.float)
            tmp_node_mask = torch.tensor(test_node_outlier_mask[i].reshape(-1, num_features_node), dtype=torch.float)

            label = torch.tensor(self.y_test[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            test_outlier_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.test_outlier_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_imputed_data = []
        for i in range(self.X_test_imputed.shape[0]):
            tmp_edge_attr = torch.tensor(test_edge_imputed_data[i].reshape(-1, num_features_edge), dtype=torch.float)
            tmp_edge_mask = torch.tensor(test_edge_imputed_mask[i].reshape(-1, num_features_edge), dtype=torch.float)

            tmp_node_attr = torch.tensor(test_node_imputed_data[i].reshape(-1, num_features_node), dtype=torch.float)
            tmp_node_mask = torch.tensor(test_node_imputed_mask[i].reshape(-1, num_features_node), dtype=torch.float)

            label = torch.tensor(self.y_test[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            test_imputed_data.append(
                Data(x=tmp_node_attr, edge_index=self.edge_index, edge_attr=tmp_edge_attr, y=label))

        self.test_imputed_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        return [edge_index, self.train_loader, self.val_loader, self.test_loader, self.test_outlier_loader, self.test_imputed_loader]

    def generate_dataset_TI_GNN_PMU_caseB(self):

        num_active_branches   = len(self.selected_edges)
        num_edges             = NUM_BRANCHES
        num_nodes             = NUM_NODES
        num_features          = 4

        #self.selected_edges = [6, 32, 9]

        train_node_data, train_node_mask = self.preprocess_data_PMU_caseB(self.X_train, self.selected_edges, num_nodes, num_features)
        val_node_data, val_node_mask   = self.preprocess_data_PMU_caseB(self.X_val, self.selected_edges, num_nodes, num_features)
        test_node_data, test_node_mask  = self.preprocess_data_PMU_caseB(self.X_test, self.selected_edges, num_nodes, num_features)

        print("X train shape: ", self.X_train.shape)
        print("X val shape: ", self.X_val.shape)
        print("X test shape: ", self.X_test.shape)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):

            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(train_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_train[i, :], dtype=torch.float)
            train_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):

            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(val_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_val[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            val_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(test_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_test[i, :], dtype=torch.float)
            # print(edge_index, edge_attr, mask, label)
            test_data.append(Data(x=tmp_node_attr, edge_index=self.edge_index, y=label))

        self.test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        return [edge_index, self.train_loader, self.val_loader, self.test_loader, None, None]

    def generate_dataset_TI_GNN_conventional(self):

        num_active_branches   = len(self.selected_edges)
        num_edges             = NUM_BRANCHES
        num_nodes             = NUM_NODES
        num_features          = 3

        #self.selected_edges = [6, 32, 9]

        train_node_data, train_node_mask = self.preprocess_data_conventional(self.X_train, self.selected_edges, num_nodes, num_features)
        val_node_data, val_node_mask   = self.preprocess_data_conventional(self.X_val, self.selected_edges, num_nodes, num_features)
        test_node_data, test_node_mask  = self.preprocess_data_conventional(self.X_test, self.selected_edges, num_nodes, num_features)

        print("X train shape: ", self.X_train.shape)
        print("X val shape: ", self.X_val.shape)
        print("X test shape: ", self.X_test.shape)

        edge_index = self.define_graph()

        # Prepare data for PyTorch Geometric with masking
        train_data = []
        for i in range(self.X_train.shape[0]):

            tmp_node_attr = torch.tensor(train_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(train_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_train[i, :], dtype=torch.float)
            #TODO Turn into edge probabilities for the sample
            topology=f"T{str(torch.argmax(label).item()+1)}"
            y_edge = [1 if j not in config_dict["IEEE33"][topology]["open_branches"] else 0 for j in range(NUM_BRANCHES)]

            train_data.append(Data(x=tmp_node_attr, mask=tmp_node_mask, edge_index=self.edge_index, y=label, y_edge=y_edge))

        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        val_data = []
        for i in range(self.X_val.shape[0]):

            tmp_node_attr = torch.tensor(val_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(val_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_val[i, :], dtype=torch.float)
            # TODO Turn into edge probabilities for the sample
            topology = f"T{str(torch.argmax(label).item() + 1)}"
            y_edge = [1 if j not in config_dict["IEEE33"][topology]["open_branches"] else 0 for j in range(NUM_BRANCHES)]

            val_data.append(Data(x=tmp_node_attr, mask=tmp_node_mask, edge_index=self.edge_index, y=label, y_edge=y_edge))

        self.val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        test_data = []
        for i in range(self.X_test.shape[0]):

            tmp_node_attr = torch.tensor(test_node_data[i].reshape(-1, num_features), dtype=torch.float)
            tmp_node_mask = torch.tensor(test_node_mask[i].reshape(-1, num_features), dtype=torch.float)

            label = torch.tensor(self.y_test[i, :], dtype=torch.float)
            # TODO Turn into edge probabilities for the sample
            topology = f"T{str(torch.argmax(label).item() + 1)}"
            y_edge = [1 if j not in config_dict["IEEE33"][topology]["open_branches"] else 0 for j in range(NUM_BRANCHES)]
            # print(edge_index, edge_attr, mask, label)
            test_data.append(Data(x=tmp_node_attr, mask=tmp_node_mask, edge_index=self.edge_index, y=label, y_edge=y_edge))

        self.test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

        return [edge_index, self.train_loader, self.val_loader, self.test_loader, None, None]

class TrainGNN_TI:

    def __init__(self, meterType, model, used_branches, train_loader, validation_loader, test_loader,
                 test_outlier_loader=None, test_imputed_loader=None):
        self.meterType          = meterType
        self.model              = model
        self.used_branches      = used_branches
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes        = NUM_TOPOLOGIES
        self.num_edges          = NUM_BRANCHES
        if self.meterType == "PMU_caseA":
            #self.model          = TI_GATWithEdgeAttrs(num_features=2, num_classes=self.num_classes, edge_attr_dim=2,
            #                                         gat_layers=7, GAT_dim=8, heads=4).to(self.device)
            self.model          = TI_TransformerWithEdges(num_features=2, num_classes=self.num_classes,
                                                          gat_layers=4, GAT_dim=12, edge_attr_dim=2, heads=4,
                                                          embedding_dim=2, dec_layers=1, ff_hid_dim=48).to(self.device)
        elif self.meterType =="PMU_caseB":
            #self.model          = TI_GATNoEdgeAttrs(num_features=4, num_classes=self.num_classes, heads=4,
            #                                        num_gat_layers=2, gat_dim=16).to(self.device)
            self.model          = TI_TransformerNoEdges(num_nodes=NUM_NODES,num_features=4,output_dim=8,
                                                        GATConv_layers=3, GATConv_dim=8, embedding_dim=4, heads=4,
                                                        dec_layers=1, ff_hid_dim=24).to(self.device)

        elif self.meterType == "conventional":
            #self.model          = TI_GATNoEdgeAttrs(num_features=3, num_classes=self.num_classes, heads=4,
            #                                        num_gat_layers=2, gat_dim=16).to(self.device)
            self.model          = TI_TransformerNoEdges(num_nodes=NUM_NODES,num_features=3,output_dim=8,
                                                        GATConv_layers=2, GATConv_dim=12, embedding_dim=4, heads=4,
                                                        dec_layers=1, ff_hid_dim=24).to(self.device)

        print(self.model)
        print("# Trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.optimizer           = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion           = nn.CrossEntropyLoss()
        self.train_loader        = train_loader
        self.validation_loader   = validation_loader
        self.test_loader         = test_loader
        self.test_outlier_loader = test_outlier_loader
        self.test_imputed_loader = test_imputed_loader

        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5, verbose=True)

    def train(self):

        # Early stopping parameters
        patience = 30  # Number of epochs to wait for improvement
        min_delta = 0.0000001  # Minimum change in validation loss to qualify as an improvement
        best_val_loss = float('inf')
        early_stop_counter = 0
        max_epochs = 300  # Maximum number of epochs to train
        best_model_weights = None  # To store the best weights

        # Training loop
        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)

                out_flat = out.view(-1, self.num_classes)
                y_flat   = batch.y.view(-1, self.num_classes)
                #y_flat = torch.tensor(batch.y_edge).float().to(self.device)


                loss = self.criterion(out_flat, y_flat)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = 0
            for batch_val in self.validation_loader:
                batch_val = batch_val.to(self.device)
                out_val = self.model(batch_val).to(self.device)

                out_flat = out_val.view(-1, self.num_classes)
                y_flat   = batch_val.y.view(-1, self.num_classes)

                loss = self.criterion(out_flat, y_flat)
                val_loss += loss.item()

            # Early stopping and best weights check
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

            print(f'Epoch {epoch + 1}, - Training Loss: {total_loss / len(self.train_loader)}', f""" - Validation Loss: {val_loss/len(self.validation_loader)}""")

        return self.model.load_state_dict(best_model_weights)

    def evaluate(self):

        self.train()
        total_loss = 0

        for batch_test in self.test_loader:
            batch_test = batch_test.to(self.device)
            out_test = self.model(batch_test)

            out_flat = out_test.view(-1, self.num_classes)
            y_flat = batch_test.y.view(-1, self.num_classes)

            loss = self.criterion(out_flat, y_flat)
            total_loss += loss.item()

        print(f""" - Evaluation (Test Set) Loss: {total_loss / len(self.test_loader)}""")

        correct_predictions = 0
        total_samples = 0

        for batch_test in self.test_loader:
            batch_test = batch_test.to(self.device)

            # Forward pass
            out_test = self.model(batch_test)

            # Flatten outputs and true labels
            out_flat = out_test.view(-1, self.num_classes)
            y_flat = batch_test.y.view(-1, self.num_classes)


            # Get predicted class by selecting the index of the maximum logit for each sample
            preds = out_flat.argmax(dim=1)
            true_labels = y_flat.argmax(dim=1)

            # Calculate the number of correct predictions
            correct_predictions += (preds == true_labels).sum().item()
            total_samples += y_flat.size(0)

        # Calculate accuracy as the proportion of correct predictions
        accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

        if self.test_outlier_loader is not None:
            correct_predictions = 0
            total_samples = 0
            for batch_test in self.test_outlier_loader:
                batch_test = batch_test.to(self.device)

                # Forward pass
                out_test = self.model(batch_test)

                # Flatten outputs and true labels
                out_flat = out_test.view(-1, self.num_classes)
                y_flat = batch_test.y.view(-1, self.num_classes)

                # Get predicted class by selecting the index of the maximum logit for each sample
                preds = out_flat.argmax(dim=1)
                true_labels = y_flat.argmax(dim=1)

                # Calculate the number of correct predictions
                correct_predictions += (preds == true_labels).sum().item()
                total_samples += y_flat.size(0)

            # Calculate accuracy as the proportion of correct predictions
            accuracy = correct_predictions / total_samples
            print(f"Test Accuracy: {accuracy:.4f}")

        if self.test_imputed_loader is not None:
            correct_predictions = 0
            total_samples = 0
            for batch_test in self.test_imputed_loader:
                batch_test = batch_test.to(self.device)

                # Forward pass
                out_test = self.model(batch_test)

                # Flatten outputs and true labels
                out_flat = out_test.view(-1, self.num_classes)
                y_flat = batch_test.y.view(-1, self.num_classes)

                # Get predicted class by selecting the index of the maximum logit for each sample
                preds = out_flat.argmax(dim=1)
                true_labels = y_flat.argmax(dim=1)

                # Calculate the number of correct predictions
                correct_predictions += (preds == true_labels).sum().item()
                total_samples += y_flat.size(0)

            # Calculate accuracy as the proportion of correct predictions
            accuracy = correct_predictions / total_samples
            print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

#TODO Adjust training for TI to PyTorch library
class TrainNN_TI:
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
        self.model         = model
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes   = num_classes

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)  # Ensure it's long for classification
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.optimizer     = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion     = nn.CrossEntropyLoss()

        print(self.model)
        print("# Trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train(self):

        # Create DataLoader for training and validation
        train_dataset = TD_NN(self.X_train, self.y_train)
        val_dataset = TD_NN(self.X_val, self.y_val)
        train_loader = DL_NN(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DL_NN(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        max_epochs = 200
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 20
        min_delta = 0.000001
        best_model_weights = None

        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:  # Process data in batches
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)  # Move to GPU if available

                self.optimizer.zero_grad()
                out = self.model(X_batch)
                loss = self.criterion(out, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)  # Multiply by batch size to get total loss

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    out_val = self.model(X_batch)
                    loss = self.criterion(out_val, y_batch)
                    val_loss += loss.item() * X_batch.size(0)  # Multiply by batch size

            # Early stopping check
            avg_val_loss = val_loss / len(val_dataset)  # Average validation loss
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                best_model_weights = self.model.state_dict()
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f'Epoch {epoch + 1}, Training Loss: {total_loss / self.X_train.shape[0]}, Validation Loss: {val_loss / self.X_val.shape[0]}')

        if best_model_weights:
            self.model.load_state_dict(best_model_weights)

    def evaluate(self):

        self.train()

        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for i in range(self.X_test.shape[0]):
                X_test_sample = self.X_test[i].unsqueeze(0).to(self.device)
                y_test_sample = self.y_test[i].unsqueeze(0).to(self.device)
                out_test = self.model(X_test_sample)
                loss = self.criterion(out_test, y_test_sample)


                # Convert to class indices if needed
                #if y_test_sample.ndimension() > 1:
                y_test_sample = torch.argmax(y_test_sample, dim=1).long()

                # Get the predicted class (class with max logit)
                preds = out_test.argmax(dim=1)
                correct_predictions += (preds == y_test_sample).sum().item()
                total_loss += loss
                total_samples += y_test_sample.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Test Loss: {total_loss / self.X_test.shape[0]}, Test Accuracy: {accuracy:.4f}")
        return accuracy



if __name__ == "__main__":

    #meterType = "PMU_caseA"
    meterType = "PMU_caseB"
    #meterType = "conventional"

    model = "NN"
    PP    = "RF"
    subPP = "rfe"
    threshold = 0.95

    PreProc = Preprocess()
    (X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs,
     y_test_labels, X_test_outliers, X_test_imputed, y_test_imputed_outputs,
     y_test_imputed_labels) = PreProc.preprocess_meter_type(meterType)
    TI_PTP = TIPredictorTrainProcess(meterType, threshold, model, X_train, y_train_labels,
                                     X_val, y_val_labels, X_test, y_test_labels, PP, subPP,
                                     X_test_outliers=X_test_outliers, X_test_imputed=X_test_imputed,
                                     y_test_imputed=y_test_imputed_labels)
    TI_PTP.execute()



    #TI_PTP.execute_NN()
    #gtip = GraphTIPreprocess_IV(meterType=meterType,
    #                            selected_edges=[32, 6, 9],
    #                            X_train=X_train,
    #                            y_train=y_train_labels,
    #                            X_val=X_val,
    #                            y_val=y_val_labels,
    #                            X_test=X_test,
    #                            y_test=y_test_labels)

    #train_node_data, train_node_mask = gtip.preprocess_data_PMU_caseB(X_train,[32, 6, 9],NUM_NODES,num_features_per_node=4)
    #print(train_node_data.shape, train_node_mask.shape)
    #edge_index, train_loader, val_loader, test_loader = gtip.generate_dataset_TI_GNN_PMU_caseB()
    #print(edge_index)
    #print(train_loader)
    #print(val_loader)
    #print(test_loader)
