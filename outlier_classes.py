import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
# Show all rows
pd.set_option('display.max_rows', None)
from config_file import NUM_BRANCHES, NUM_NODES, branch_data

class OutlierInjector:

    def __init__(self, X_dataset):

        self.X_dataset = X_dataset

    # Generates outliers per node sample
    def generate_outlier_value(self, normal_value, type="voltage_magnitude"):
        # Voltage normally ~1.0 p.u., outliers could be severe drops or spikes
        if type in ["voltage_magnitude", "current_magnitude"]:
            return normal_value * np.random.choice([0.1, 10])
        elif type in ["voltage_angle"]:
            return normal_value + 20 if normal_value>0 else normal_value - 20
        #elif type in ["current_angle"]:
        #    return normal_value + 20 if normal_value>0 else normal_value - 20
        else:
            return normal_value

    def outlier_event(self, probability=0.01):
        return random.random() < probability

    # Inject outliers per input sample at specified indices
    def inject_outliers_PMUcaseA(self, X_dataset, injection_edge_indices, prob_m=0.01, prob_a=0.001):

        outlier_index_dict = {
            "Vm":  [],
            "Va":  [],
            "Im":  [],
            "ALL": [],
        }
        outlier_num = 0
        outlier_data_list = []

        for sample_index in range(X_dataset.shape[0]):
            Vm_list = list(X_dataset[sample_index][0:NUM_NODES])
            Va_list = list(X_dataset[sample_index][NUM_NODES:2*NUM_NODES])
            Im_list = list(X_dataset[sample_index][2*NUM_NODES:2*NUM_NODES+NUM_BRANCHES])
            Ia_list = list(X_dataset[sample_index][2*NUM_NODES+NUM_BRANCHES:])
            flag=False
            for edge_idx in injection_edge_indices:
                sending_node_idx = branch_data[edge_idx]["sending_node"]
                if self.outlier_event(probability=prob_m):
                    Im = Im_list[sending_node_idx]
                    Im_outlier_value = self.generate_outlier_value(Im, "current_magnitude")
                    Im_list[edge_idx] = Im_outlier_value
                    outlier_num+=1
                    outlier_index_dict["Im"].append(sample_index)
                if self.outlier_event(probability=prob_m):
                    Vm = Vm_list[sending_node_idx]
                    Vm_outlier_value = self.generate_outlier_value(Vm, "voltage_magnitude")
                    Vm_list[sending_node_idx] = Vm_outlier_value
                    flag = True
                    outlier_num+=1
                    outlier_index_dict["Vm"].append(sample_index)
                if self.outlier_event(probability=prob_m):
                    Va = Va_list[sending_node_idx]
                    Va_outlier_value = self.generate_outlier_value(Va, "voltage_angle")
                    Va_list[sending_node_idx] = Va_outlier_value
                    outlier_num+=1
                    outlier_index_dict["Va"].append(sample_index)
                if self.outlier_event(probability=prob_m):
                    Im = Im_list[edge_idx]
                    Vm = Vm_list[sending_node_idx]
                    Va = Va_list[sending_node_idx]

                    Im_outlier_value = self.generate_outlier_value(Im, "current_magnitude")
                    Vm_outlier_value = self.generate_outlier_value(Vm, "voltage_magnitude")
                    Va_outlier_value = self.generate_outlier_value(Va, "voltage_angle")

                    Ia_list[edge_idx] = Im_outlier_value
                    Vm_list[sending_node_idx] = Vm_outlier_value
                    Va_list[sending_node_idx] = Va_outlier_value

                    outlier_num+=1
                    outlier_index_dict["ALL"].append(sample_index)


            sample = Vm_list + Va_list + Im_list + Ia_list

            outlier_data_list.append(sample)

        outlier_dataset = np.array(outlier_data_list)

        return outlier_dataset, outlier_index_dict

class IF_OutlierDetection:

    def __init__(self, X_arr, outlier_index_dict, meter):
        self.X_arr              = X_arr
        self.outlier_index_dict = outlier_index_dict
        self.model_Im           = IsolationForest(contamination=0.02, random_state=42)
        self.model_Vm           = IsolationForest(contamination=0.02, random_state=42)
        self.model_Va           = IsolationForest(contamination=0.02, random_state=42)
        self.model_ALL          = IsolationForest(contamination=0.01, random_state=42)
        self.meter              = meter

    def train_and_evaluate_isolation_forest_PMU_caseA(self, X_train, X_test):

        per_edge_outliers = {
            "Im":  [],
            "Vm":  [],
            "Va":  [],
            "ALL": [],
        }

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        #TODO Train Isolation Forest - ALL
        # Keep only meter relevant information and predict per meter
        dataset = X_train[:, [node_idx, NUM_NODES+node_idx, 2*NUM_NODES+edge_idx, 2*NUM_NODES+NUM_BRANCHES+edge_idx]]
        self.model_ALL.fit(dataset)

        #TODO Im
        # Keep only meter relevant information and predict per meter
        dataset = X_train[:, [2*NUM_NODES+edge_idx]]
        self.model_Im.fit(dataset)

        #TODO Vm
        # Keep only meter relevant information and predict per meter
        dataset = X_train[:, [node_idx]]
        self.model_Vm.fit(dataset)

        #TODO Va
        # Keep only meter relevant information and predict per meter
        dataset = X_train[:, [NUM_NODES + node_idx]]
        self.model_Va.fit(dataset)


    def divide_outlier_indices(self, per_edge_outliers):

        per_edge_outliers["Im"] = sorted(list(set(per_edge_outliers["Im"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Im outliers: ", len(per_edge_outliers["Im"]))
        per_edge_outliers["Vm"] = sorted(list(set(per_edge_outliers["Vm"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Vm outliers: ", len(per_edge_outliers["Vm"]))
        per_edge_outliers["Va"] = sorted(list(set(per_edge_outliers["Va"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Va outliers: ", len(per_edge_outliers["Va"]))

        return per_edge_outliers


    def predict_outliers_PMU_caseA(self, X_test):
        per_edge_outliers = {
            "Im": [],
            "Vm": [],
            "Va": [],
            "ALL": [],
        }

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        # TODO ALL
        # Keep only meter relevant information and predict per meter
        dataset = X_test[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]
        predictions = self.model_ALL.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = self.outlier_index_dict["ALL"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR ALL MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
              "Real Outliers: ", len(real_outliers),
              "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["ALL"] = list(predicted_outlier_indices)

        # TODO Im
        dataset = X_test[:, [2 * NUM_NODES + edge_idx]]
        predictions = self.model_Im.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = self.outlier_index_dict["Im"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Im MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
              "Real Outliers: ", len(real_outliers),
              "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Im"] = list(predicted_outlier_indices)

        # TODO Vm
        dataset = X_test[:, [node_idx]]
        predictions = self.model_Vm.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = self.outlier_index_dict["Vm"]

        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Vm MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
              "Real Outliers: ", len(real_outliers),
              "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Vm"] = list(predicted_outlier_indices)

        # TODO Va
        # Keep only meter relevant information and predict per meter
        dataset = X_test[:, [NUM_NODES + node_idx]]
        predictions = self.model_Va.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = self.outlier_index_dict["Va"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Va MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%",
              "Real Outliers: ", len(real_outliers),
              "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Va"] = list(predicted_outlier_indices)

        per_edge_outliers = self.divide_outlier_indices(per_edge_outliers)

        return per_edge_outliers

class KNNImputerMeasurements:

    def __init__(self, X_dataset, outlier_index_dict, meter):
        self.X_dataset          = X_dataset
        self.KNNImputer         = KNNImputer(n_neighbors=10)
        self.outlier_index_dict = outlier_index_dict
        self.meter              = meter

    def train_KNNImputers_PMU_caseA(self, X_train):

        outlier_indices = list(set(self.outlier_index_dict["Vm"] +
                                   self.outlier_index_dict["Va"] +
                                   self.outlier_index_dict["Im"] +
                                   self.outlier_index_dict["ALL"]))

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        inliners = X_train[~np.isin(np.arange(len(self.X_dataset)), outlier_indices)]
        print(inliners.shape)

        inliners = inliners[:, [node_idx, NUM_NODES+node_idx, 2*NUM_NODES+edge_idx, 2*NUM_NODES+NUM_BRANCHES+edge_idx]]
        self.KNNImputer.fit(inliners)


    def impute_valeus_PMU_caseA(self, compare_np):

        X_imputed = self.X_dataset.copy()

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        # TODO Vm
        X_train    = self.X_dataset[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]
        compare_np = compare_np[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]

        # Vm
        outlier_indices = self.outlier_index_dict["Vm"]
        X_train[outlier_indices, 0] = np.nan
        X_train_Vm = self.KNNImputer.transform(X_train)
        original_Vm = compare_np[outlier_indices, 0].flatten().tolist()
        imputed_Vm  = X_train_Vm[outlier_indices, 0].flatten().tolist()
        mape = sum([abs(original_Vm[i] - imputed_Vm[i]) / original_Vm[i] for i in range(len(original_Vm))])/len(original_Vm)
        print("MAPE Imputed Vm: ", mape, " - Imputed values: ", len(imputed_Vm))

        # TODO Va
        outlier_indices = self.outlier_index_dict["Va"]
        X_train[outlier_indices, 1] = np.nan
        X_train_Va = self.KNNImputer.transform(X_train)
        original_Va = compare_np[outlier_indices, 1].flatten().tolist()
        imputed_Va = X_train_Va[outlier_indices, 1].flatten().tolist()
        mae = sum([abs(original_Va[i] - imputed_Va[i]) for i in range(len(original_Va))]) / len(original_Va)
        print("MAE Imputed Va: ", mae, " - Imputed values: ", len(imputed_Va))

        # TODO Im
        outlier_indices = self.outlier_index_dict["Im"]
        X_train[outlier_indices, 2] = np.nan
        X_train_Im = self.KNNImputer.transform(X_train)
        original_Im = compare_np[outlier_indices, 2].flatten().tolist()
        imputed_Im = X_train_Im[outlier_indices, 2].flatten().tolist()

        mape = sum([abs(original_Im[i] - imputed_Im[i]) / original_Im[i] for i in range(len(original_Im))]) / len(original_Im)
        print("MAPE Imputed Im: ", mape, " - Imputed values: ", len(imputed_Im))

        # Impute values on original frame X_dataset
        Im_outlier_indices = self.outlier_index_dict["Im"]
        Vm_outlier_indices = self.outlier_index_dict["Vm"]
        Va_outlier_indices = self.outlier_index_dict["Va"]

        X_imputed[Im_outlier_indices, 2*NUM_NODES + edge_idx]   = imputed_Im
        X_imputed[Vm_outlier_indices, node_idx]                 = imputed_Vm
        X_imputed[Va_outlier_indices, NUM_NODES + node_idx]     = imputed_Va

        return X_imputed

if __name__ == "__main__":

    meter = 75

    # TODO Conventional
    # X_train = np.load("datasets/outlier_datasets/95UKGD_conventional_X_train.npy")
    X_train = np.load("datasets/outlier_datasets/95UKGDPMU_caseA_input.npy")
    X_old   = X_train.copy()

    # Outlier injection into the datasets
    out_inj = OutlierInjector(X_train)
    outlier_arr, outlier_index_dict = out_inj.inject_outliers_PMUcaseA(X_train, [meter])

    #TODO Isolation Forest Outlier Detection - Gets outlier X_train dataset as input
    iso_detector = IF_OutlierDetection(outlier_arr, outlier_index_dict, meter)
    iso_detector.train_and_evaluate_isolation_forest_PMU_caseA(outlier_arr, outlier_arr)
    per_edge_outliers = iso_detector.predict_outliers_PMU_caseA(outlier_arr)

    # KNN imputer on imputer values
    knn_imp = KNNImputerMeasurements(outlier_arr, per_edge_outliers, meter)
    knn_imp.train_KNNImputers_PMU_caseA(outlier_arr)
    X_test = knn_imp.impute_valeus_PMU_caseA(compare_np=X_old)

