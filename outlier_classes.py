import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from config_file import NUM_BRANCHES, NUM_NODES, branch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config_file import *
# Show all rows

class BadDataInjector:

    def __init__(self, meterType, feature_list):
        self.input          = f"datasets/{dataset}_PMU_caseA_input.npy"
        self.output         = f"datasets/{dataset}_PMU_caseA_output.npy"
        self.meterType      = meterType
        self.feature_list   = feature_list
        self.prob_m         = 0.002
        self.prob_a         = 0.002
        self.X_train_orig   = None
        self.y_train_orig   = None
        self.X_val_orig     = None
        self.y_val_orig     = None
        self.X_test_orig    = None
        self.y_test_orig    = None


    def custom_one_hot_encode(self, labels):

        one_hot_encoded_labels = []
        for label in labels:
            tmp = [0 for i in range(NUM_TOPOLOGIES)]
            tmp[int(label)-1] = 1

            one_hot_encoded_labels.append(tmp)

        one_hot_encoded_labels = np.array(one_hot_encoded_labels)

        return one_hot_encoded_labels

    def outlier_event(self, probability=0.02):
        return random.random() < probability

    def generate_outlier_value(self, normal_value, type="voltage_magnitude"):
        # Voltage normally ~1.0 p.u., outliers could be severe drops or spikes
        if type in ["voltage_magnitude"]:
            return normal_value * np.random.choice([0.05, 20])
        elif type in ["current_magnitude"]:
            return normal_value * np.random.choice([20])
        elif type in ["voltage_angle"]:
            return normal_value + 40 if normal_value>0 else normal_value - 40
        else:
            return normal_value

    def inject_outliers_PMU_caseA(self, X, y, note=""):

        X_outlier_dict = {
            i: {
                "Vm": [],
                "Va": [],
                "Im": [],
            } for i in self.feature_list
        }
        outlier_num = 0
        outlier_data_list = []

        for sample_index in range(X.shape[0]):
            Vm_list = list(X[sample_index][0:NUM_NODES])
            Va_list = list(X[sample_index][1*NUM_NODES:2*NUM_NODES])
            Im_list = list(X[sample_index][2*NUM_NODES:2*NUM_NODES+NUM_BRANCHES])
            Ia_list = list(X[sample_index][2*NUM_NODES+NUM_BRANCHES:])

            for edge_idx in self.feature_list:
                sending_node_idx = branch_data[edge_idx]["sending_node"]
                if self.outlier_event(probability=self.prob_m) and Im_list[edge_idx]>0:
                    Im = Im_list[edge_idx]
                    Im_outlier_value = self.generate_outlier_value(Im, "current_magnitude")
                    Im_list[edge_idx] = Im_outlier_value

                    outlier_num += 1
                    X_outlier_dict[edge_idx]["Im"].append(sample_index)
                if self.outlier_event(probability=self.prob_m):
                    Vm = Vm_list[sending_node_idx]
                    Vm_outlier_value = self.generate_outlier_value(Vm, "voltage_magnitude")
                    Vm_list[sending_node_idx] = Vm_outlier_value
                    flag = True
                    outlier_num += 1
                    X_outlier_dict[edge_idx]["Vm"].append(sample_index)
                if self.outlier_event(probability=self.prob_m):
                    Va = Va_list[sending_node_idx]
                    Va_outlier_value = self.generate_outlier_value(Va, "voltage_angle")
                    Va_list[sending_node_idx] = Va_outlier_value
                    outlier_num += 1
                    X_outlier_dict[edge_idx]["Va"].append(sample_index)

            sample = Vm_list + Va_list + Im_list + Ia_list
            outlier_data_list.append(sample)

        print("Outlier number: ", outlier_num)
        X = np.array(outlier_data_list)

        if note=="test":
            for sensor_idx in X_outlier_dict.keys():
                    for key in X_outlier_dict[sensor_idx].keys():
                        print(f"""{sensor_idx}-{key} bad-data length: , {str(len(X_outlier_dict[sensor_idx][key]))}""")

        return X, y, X_outlier_dict

    def inject_outliers_PMU_caseB(self, X, y):

        X_outlier_dict = {
            i: {
                "Vm": [],
                "Va": [],
                "Im": [],
            } for i in self.feature_list
        }
        outlier_num = 0
        outlier_data_list = []

        for sample_index in range(X.shape[0]):
            Vm_list = list(X[sample_index][0:NUM_NODES])
            Va_list = list(X[sample_index][1*NUM_NODES:2*NUM_NODES])
            Im_list = list(X[sample_index][2*NUM_NODES:3*NUM_NODES])
            Ia_list = list(X[sample_index][3*NUM_NODES:])

            for node_idx in self.feature_list:
                if self.outlier_event(probability=self.prob_m):
                    Im = Im_list[node_idx]
                    Im_outlier_value = self.generate_outlier_value(Im, "current_magnitude")
                    Im_list[node_idx] = Im_outlier_value
                    outlier_num += 1
                    X_outlier_dict[node_idx]["Im"].append(sample_index)
                if self.outlier_event(probability=self.prob_m):
                    Vm = Vm_list[node_idx]
                    Vm_outlier_value = self.generate_outlier_value(Vm, "voltage_magnitude")
                    Vm_list[node_idx] = Vm_outlier_value
                    outlier_num += 1
                    X_outlier_dict[node_idx]["Vm"].append(sample_index)
                if self.outlier_event(probability=self.prob_m):
                    Va = Va_list[node_idx]
                    Va_outlier_value = self.generate_outlier_value(Va, "voltage_angle")
                    Va_list[node_idx] = Va_outlier_value
                    outlier_num += 1
                    X_outlier_dict[node_idx]["Va"].append(sample_index)


            sample = Vm_list + Va_list + Im_list + Ia_list
            outlier_data_list.append(sample)

        X = np.array(outlier_data_list)

        return X, y, X_outlier_dict

    def generate_datasets(self):

        inputs = np.load(self.input)
        print("Input shape: ", inputs.shape)

        outputs = np.load(self.output)

        # Reshape the labels to a 2D array
        output_SE, output_TI = outputs[:, :-1], outputs[:, -1]

        labels_reshaped = list(output_TI)

        ohe_labels = self.custom_one_hot_encode(labels_reshaped)

        # Initialize an empty list to store concatenated elements
        concatenated_list = []

        # Loop through each element (i) of outputs and labels
        for i in range(len(outputs)):
            # Concatenate the ith elements of outputs and labels
            concatenated = np.concatenate((output_SE[i], ohe_labels[i]))
            concatenated_list.append(concatenated)

        outputs = np.array(concatenated_list)

        print("Output shape: ", outputs.shape, "Output_SE shape: ", output_SE.shape, "Output_TI shape: ", output_TI.shape)

        # First split: train validation and test
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.15, random_state=42)
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)  # 0.25 x 0.8 = 0.2

        self.X_train_orig = X_train.copy()
        self.y_train_orig = y_train.copy()
        self.X_val_orig   = X_val.copy()
        self.y_val_orig   = y_val.copy()
        self.X_test_orig  = X_test.copy()
        self.y_test_orig  = y_test.copy()

        if self.meterType == "PMU_caseA":
            X_train, y_train, X_train_indices = self.inject_outliers_PMU_caseA(X_train, y_train)
            X_val, y_val, X_val_indices       = self.inject_outliers_PMU_caseA(X_val, y_val)
            X_test, y_test, X_test_indices    = self.inject_outliers_PMU_caseA(X_test, y_test, note="test")
        elif self.meterType == "PMU_caseB":
            X_train, y_train, X_train_indices = self.inject_outliers_PMU_caseB(X_train, y_train)
            X_val, y_val, X_val_indices = self.inject_outliers_PMU_caseB(X_val, y_val)
            X_test, y_test, X_test_indices = self.inject_outliers_PMU_caseB(X_test, y_test)
        else:
            X_train, y_train, X_train_indices, X_val, y_val, X_val_indices, \
                X_test, y_test, X_test_indices = None, None, None, None, None, None, None, None, None

        return X_train, y_train, X_train_indices, X_val, y_val, X_val_indices, X_test, y_test, X_test_indices, \
            self.X_train_orig, self.y_train_orig, self.X_val_orig, self.y_val_orig, self.X_test_orig, self.y_test_orig

class IF_BadDataDetector:

    def __init__(self, meterType, feature_list, X_train, y_train, X_train_outlier_indices,
                 X_val, y_val, X_val_outlier_indices, X_test, y_test, X_test_outlier_indices):
        self.feature_list               = feature_list
        self.meterType                  = meterType
        self.X_train                    = X_train
        self.y_train                    = y_train
        self.X_train_outlier_indices    = X_train_outlier_indices
        self.X_val                      = X_val
        self.y_val                      = y_val
        self.X_val_outlier_indices      = X_val_outlier_indices
        self.X_test                     = X_test
        self.y_test                     = y_test
        self.X_test_outlier_indices     = X_test_outlier_indices
        # Isolation Forest dictionary for every node-feature combination
        self.IF_dict = {
            i: {
                "Vm": IsolationForest(max_samples=256, contamination=0.01, random_state=42),
                "Va": IsolationForest(max_samples=256, contamination=0.01, random_state=42),
                "Im": IsolationForest(max_samples=256, contamination=0.01, random_state=42),
            } for i in self.feature_list
        }
        self.predicted_outliers = {
            i: {
                "Vm": [],
                "Va": [],
                "Im": [],
            } for i in self.feature_list
        }
        # Predicted outliers dictionary
        self.predicted_outliers_val = {
            i: {
                "Vm": [],
                "Va": [],
                "Im": [],
            } for i in self.feature_list
        }
        # Predicted outliers dictionary
        self.predicted_outliers_test = {
            i: {
                "Vm": [],
                "Va": [],
                "Im": [],
            } for i in self.feature_list
        }

    def train_IF(self):

        #TODO Train IFs

        if self.meterType == "PMU_caseA":
            for branch_idx in self.feature_list:
                for feature in ["Vm", "Va", "Im"]:
                    node_idx = branch_data[branch_idx]["sending_node"]
                    if feature == "Vm":
                        X_tmp = self.X_train[:, 0*NUM_NODES+node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Vm"]
                        iso_f.fit(X_tmp)
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_train_outlier_indices[branch_idx]["Vm"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Vm MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))

                        self.predicted_outliers[branch_idx]["Vm"] = predicted_outlier_indices

                    elif feature == "Va":
                        X_tmp = self.X_train[:,1*NUM_NODES+node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Va"]
                        iso_f.fit(X_tmp)
                        predictions = iso_f.predict(X_tmp)

                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_train_outlier_indices[branch_idx]["Va"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Va MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.predicted_outliers[branch_idx]["Va"] = predicted_outlier_indices

                    elif feature == "Im":
                        X_tmp = self.X_train[:,2*NUM_NODES+branch_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Im"]
                        iso_f.fit(X_tmp)
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_train_outlier_indices[branch_idx]["Im"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Im MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.predicted_outliers[branch_idx]["Im"] = predicted_outlier_indices


        elif self.meterType == "PMU_caseB":
            for node_idx in self.feature_list:
                for feature in ["Vm", "Va", "Im"]:
                    if feature == "Vm": X_tmp = self.X_train[:, node_idx]
                    elif feature == "Va": X_tmp = self.X_train[:, NUM_NODES+node_idx]
                    elif feature == "Im": X_tmp = self.X_train[:, 2*NUM_NODES+node_idx]

    def predict(self):

        if self.meterType == "PMU_caseA":
            for branch_idx in self.feature_list:
                for feature in ["Vm", "Va", "Im"]:
                    node_idx = branch_data[branch_idx]["sending_node"]
                    #TODO Validation - Testing sets
                    if feature == "Vm":
                        X_tmp = self.X_val[:, node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Vm"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_val_outlier_indices[branch_idx]["Vm"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Vm -validation- MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.X_val_outlier_indices[branch_idx]["Vm"] = predicted_outlier_indices

                        X_tmp = self.X_test[:, node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Vm"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_test_outlier_indices[branch_idx]["Vm"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Vm -test- MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.X_test_outlier_indices[branch_idx]["Vm"] = predicted_outlier_indices

                    elif feature == "Va":
                        X_tmp = self.X_val[:, NUM_NODES+node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Va"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_val_outlier_indices[branch_idx]["Va"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Va -validation- MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.X_val_outlier_indices[branch_idx]["Va"] = predicted_outlier_indices

                        X_tmp = self.X_test[:, NUM_NODES+node_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Va"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_test_outlier_indices[branch_idx]["Va"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Va -test- MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
                        self.X_test_outlier_indices[branch_idx]["Va"] = predicted_outlier_indices

                    elif feature == "Im":
                        X_tmp = self.X_val[:, 2*NUM_NODES + branch_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Im"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_val_outlier_indices[branch_idx]["Im"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Im -validation- MEASUREMENTS ERRORS: ",
                              100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:",
                              len(predicted_outlier_indices))
                        self.X_val_outlier_indices[branch_idx]["Im"] = predicted_outlier_indices

                        X_tmp = self.X_test[:, 2*NUM_NODES + branch_idx].reshape(-1, 1)
                        iso_f = self.IF_dict[branch_idx]["Im"]
                        predictions = iso_f.predict(X_tmp)
                        predicted_outlier_indices = list(np.where(predictions == -1)[0])
                        real_outliers = self.X_test_outlier_indices[branch_idx]["Im"]
                        set_diff = set(real_outliers) - set(predicted_outlier_indices)
                        print(f"{branch_idx}-NOT FOUND FOR Im -test- MEASUREMENTS ERRORS: ",
                              100 * (len(set_diff) / len(real_outliers)), "%",
                              "Real Outliers: ", len(real_outliers), "Predicted Outliers:",
                              len(predicted_outlier_indices))
                        self.X_test_outlier_indices[branch_idx]["Im"] = predicted_outlier_indices

        return self.X_val_outlier_indices, self.X_test_outlier_indices

class KNN_Imputer:

    def __init__(self, meterType, feature_list,
                 X_train, y_train, X_train_indices,
                 X_val, y_val, X_val_indices,
                 X_test, y_test, X_test_indices,
                 orig_datasets):
        self.X_train            = X_train
        self.y_train            = y_train
        self.X_val              = X_val
        self.y_val              = y_val
        self.X_test             = X_test
        self.y_test             = y_test
        self.X_train_indices    = X_train_indices
        self.X_val_indices      = X_val_indices
        self.X_test_indices     = X_test_indices
        self.features_list      = feature_list
        self.meterType          = meterType
        self.X_train_orig, self.y_train_orig, self.X_val_orig, self.y_val_orig, \
            self.X_test_orig, self.y_test_orig = orig_datasets
        self.X_train_imputed, self.X_val_imputed, self.X_test_imputed = self.X_train.copy(), \
            self.X_val.copy(), self.X_test.copy()

        if self.meterType == "PMU_caseA":
            self.knn_dict = {
                i: {
                    "Vm": KNNImputer(n_neighbors=10),
                    "Va": KNNImputer(n_neighbors=10),
                    "Im": KNNImputer(n_neighbors=10),

                } for i in self.features_list
            }


    def fit_knns(self):

        for edge_idx in self.features_list:
            if self.meterType == "PMU_caseA":
                sending_node_idx = branch_data[edge_idx]["sending_node"]
                feature_indices = [sending_node_idx, NUM_NODES+sending_node_idx, 2*NUM_NODES+edge_idx, 2*NUM_NODES+NUM_BRANCHES+edge_idx]
                for feature in ["Vm", "Va", "Im"]:
                    if feature == "Vm":
                        X_tmp = self.X_train[:, feature_indices].copy()
                        outlier_indices = self.X_train_indices[edge_idx]["Vm"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Vm"].fit(inliners)
                        X_tmp[outlier_indices, 0] = np.nan
                        X_train_Vm = self.knn_dict[edge_idx]["Vm"].transform(X_tmp)
                        original   = self.X_train_orig[outlier_indices, sending_node_idx]
                        imputed    = X_train_Vm[outlier_indices, 0]
                        mape = np.mean(np.abs((original - imputed) / original))
                        print(f"MAPE for {edge_idx}-Vm", mape)
                        self.X_train_imputed[outlier_indices, 0] = X_train_Vm[outlier_indices, 0]

                    elif feature == "Va":
                        X_tmp = self.X_train[:, feature_indices].copy()
                        outlier_indices = self.X_train_indices[edge_idx]["Va"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Va"].fit(inliners)
                        X_tmp[outlier_indices, 1] = np.nan
                        X_train_Va = self.knn_dict[edge_idx]["Va"].transform(X_tmp)
                        original   = self.X_train_orig[outlier_indices, NUM_NODES+sending_node_idx]
                        imputed    = X_train_Va[outlier_indices, 1]
                        mae = np.mean(np.abs((original - imputed) / len(original)))
                        print(f"MAE for {edge_idx}-Va", mae)
                        self.X_train_imputed[outlier_indices, 1] = X_train_Va[outlier_indices, 1]

                    elif feature == "Im":
                        X_tmp = self.X_train[:, feature_indices].copy()
                        outlier_indices = self.X_train_indices[edge_idx]["Im"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Im"].fit(inliners)
                        X_tmp[outlier_indices, 2] = np.nan
                        X_train_Im = self.knn_dict[edge_idx]["Im"].transform(X_tmp)
                        original   = self.X_train_orig[outlier_indices, 2*NUM_NODES+edge_idx]
                        imputed    = X_train_Im[outlier_indices, 2]
                        # Example: original and imputed are NumPy arrays
                        mask = original != 0  # Only keep non-zero original values
                        # Apply MAPE formula on valid entries
                        mape = np.mean(np.abs((original[mask] - imputed[mask]) / original[mask]))
                        print(f"MAPE for {edge_idx}-Im", mape)
                        self.X_train_imputed[outlier_indices, 2] = X_train_Im[outlier_indices, 2]

    def impute_test_set(self):

        for edge_idx in self.features_list:
            if self.meterType == "PMU_caseA":
                sending_node_idx = branch_data[edge_idx]["sending_node"]
                feature_indices = [sending_node_idx, NUM_NODES+sending_node_idx, 2*NUM_NODES+edge_idx, 2*NUM_NODES+NUM_BRANCHES+edge_idx]
                for feature in ["Vm", "Va", "Im"]:
                    if feature == "Vm":
                        X_tmp = self.X_test[:, feature_indices].copy()
                        outlier_indices = self.X_test_indices[edge_idx]["Vm"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Vm"].fit(inliners)
                        X_tmp[outlier_indices, 0] = np.nan
                        X_test_Vm = self.knn_dict[edge_idx]["Vm"].transform(X_tmp)
                        original   = self.X_test_orig[outlier_indices, sending_node_idx]
                        imputed    = X_test_Vm[outlier_indices, 0]
                        mape = np.mean(np.abs((original - imputed) / original))
                        print(f"MAPE for {edge_idx}-Vm", mape)
                        self.X_test_imputed[outlier_indices, sending_node_idx] = X_test_Vm[outlier_indices, 0]

                    elif feature == "Va":
                        X_tmp = self.X_test[:, feature_indices].copy()
                        outlier_indices = self.X_test_indices[edge_idx]["Va"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Va"].fit(inliners)
                        X_tmp[outlier_indices, 1] = np.nan
                        X_test_Va = self.knn_dict[edge_idx]["Va"].transform(X_tmp)
                        original   = self.X_test_orig[outlier_indices, NUM_NODES+sending_node_idx]
                        imputed    = X_test_Va[outlier_indices, 1]
                        mae = np.mean(np.abs((original - imputed) / len(original)))
                        print(f"MAE for {edge_idx}-Va", mae)
                        self.X_test_imputed[outlier_indices, NUM_NODES+sending_node_idx] = X_test_Va[outlier_indices, 1]

                    elif feature == "Im":
                        X_tmp = self.X_test[:, feature_indices].copy()
                        outlier_indices = self.X_test_indices[edge_idx]["Im"]
                        inliners = X_tmp[~np.isin(np.arange(len(X_tmp)), outlier_indices)]
                        self.knn_dict[edge_idx]["Im"].fit(inliners)
                        X_tmp[outlier_indices, 2] = np.nan
                        X_test_Im = self.knn_dict[edge_idx]["Im"].transform(X_tmp)
                        original   = self.X_test_orig[outlier_indices, 2*NUM_NODES+edge_idx]
                        imputed    = X_test_Im[outlier_indices, 2]
                        # Example: original and imputed are NumPy arrays
                        mask = original != 0  # Only keep non-zero original values
                        # Apply MAPE formula on valid entries
                        mape = np.mean(np.abs((original[mask] - imputed[mask]) / original[mask]))
                        print(f"MAPE for {edge_idx}-Im", mape)
                        self.X_test_imputed[outlier_indices, 2*NUM_NODES+edge_idx] = X_test_Im[outlier_indices, 2]

        return self.X_test_imputed

if __name__ == "__main__":

    meterType = "PMU_caseA"

    bdi = BadDataInjector(meterType=meterType,feature_list=[32, 27, 7, 3])
    X_train_outliers, y_train, X_train_indices, X_val_outliers, y_val, X_val_indices, X_test_outliers, y_test, \
        X_test_indices, X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig = bdi.generate_datasets()

    bdd = IF_BadDataDetector(meterType=meterType,
                             feature_list=[32, 27, 7, 3],
                             X_train=X_train_outliers,
                             y_train=y_train,
                             X_train_outlier_indices=X_train_indices,
                             X_val=X_val_outliers,
                             y_val=y_val,
                             X_val_outlier_indices=X_val_indices,
                             X_test=X_test_outliers,
                             y_test=y_test,
                             X_test_outlier_indices=X_test_indices)
    bdd.train_IF()
    X_val_outlier_indices, X_test_outlier_indices = bdd.predict()
    print("X_test indices: ", X_test_outlier_indices)

    ki = KNN_Imputer(meterType=meterType,
                             feature_list=[32, 27, 7, 3],
                             X_train=X_train_outliers,
                             y_train=y_train,
                             X_train_indices=X_train_indices,
                             X_val=X_val_outliers,
                             y_val=y_val,
                             X_val_indices=X_val_indices,
                             X_test=X_test_outliers,
                             y_test=y_test,
                             X_test_indices=X_test_indices,
                             orig_datasets = [X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig])
    ki.fit_knns()
    X_test_imputed = ki.impute_test_set()

    #TODO scale values
    scaler = StandardScaler()
    X_train_orig      = scaler.fit_transform(X_train_orig)
    X_val_orig        = scaler.transform(X_val_orig)
    X_test_orig       = scaler.transform(X_test_orig)
    X_test_outliers  = scaler.transform(X_test_outliers)
    X_test_imputed    = scaler.transform(X_test_imputed)

    #TODO Files to store
    np.save(X_train_PMU_caseA, X_train_orig)
    np.save(y_train_PMU_caseA, y_train_orig)
    np.save(X_val_PMU_caseA, X_val_orig)
    np.save(y_val_PMU_caseA, y_val_orig)
    np.save(X_test_PMU_caseA, X_test_orig)
    np.save(y_test_PMU_caseA, y_test_orig)
    np.save(X_test_PMU_caseA_outliers, X_test_outliers)
    np.save(X_test_PMU_caseA_imputed, X_test_imputed)
    print(X_test_outliers.shape)

    #X_val_PMU_caseA
    #y_val_PMU_caseA
    #X_test_PMU_caseA
    #X_test_PMU_caseA_imputed
    #y_test_PMU_caseA






