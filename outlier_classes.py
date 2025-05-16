import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from config_file import NUM_BRANCHES, NUM_NODES, branch_data
from sklearn.model_selection import train_test_split
# Show all rows
pd.set_option('display.max_rows', None)

class OutlierInjector:

    def __init__(self, X_dataset):

        self.X_dataset = X_dataset

    def outlier_event(self, probability=0.01):
        return random.random() < probability

    # Generates outliers per node sample
    def generate_outlier_value(self, normal_value, type="voltage_magnitude"):
        # Voltage normally ~1.0 p.u., outliers could be severe drops or spikes
        if type in ["voltage_magnitude"]:
            return normal_value * np.random.choice([0.05, 20])
        elif type in ["current_magnitude"]:
            return normal_value * np.random.choice([0.05, 20])
        elif type in ["voltage_angle"]:
            return normal_value + 40 if normal_value>0 else normal_value - 40
        #elif type in ["current_angle"]:
        #    return normal_value + 20 if normal_value>0 else normal_value - 20
        else:
            return normal_value


    # Inject outliers per input sample at specified indices
    def inject_outliers_PMUcaseA(self, injection_edge_indices, prob_m=0.001, prob_a=0.001):

        outlier_index_dict = {
            "Vm":  [],
            "Va":  [],
            "Im":  [],
            "ALL": [],
        }
        outlier_num = 0
        outlier_data_list = []

        for sample_index in range(self.X_dataset.shape[0]):
            Vm_list = list(self.X_dataset[sample_index][0:NUM_NODES])
            Va_list = list(self.X_dataset[sample_index][NUM_NODES:2*NUM_NODES])
            Im_list = list(self.X_dataset[sample_index][2*NUM_NODES:2*NUM_NODES+NUM_BRANCHES])
            Ia_list = list(self.X_dataset[sample_index][2*NUM_NODES+NUM_BRANCHES:])

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
                if self.outlier_event(probability=prob_a):
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

    def __init__(self, meter):
        self.model_Im           = IsolationForest(max_samples=256, contamination=0.02, random_state=42)
        self.model_Vm           = IsolationForest(max_samples=256, contamination=0.02, random_state=42)
        self.model_Va           = IsolationForest(max_samples=256, contamination=0.02, random_state=42)
        self.model_ALL          = IsolationForest(max_samples=256, contamination=0.01, random_state=42)
        self.meter              = meter

    def divide_outlier_indices(self, per_edge_outliers):

        per_edge_outliers["Im"] = sorted(list(set(per_edge_outliers["Im"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Im outliers: ", len(per_edge_outliers["Im"]))
        per_edge_outliers["Vm"] = sorted(list(set(per_edge_outliers["Vm"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Vm outliers: ", len(per_edge_outliers["Vm"]))
        per_edge_outliers["Va"] = sorted(list(set(per_edge_outliers["Va"]) - set(per_edge_outliers["ALL"])))
        print("Standalone Va outliers: ", len(per_edge_outliers["Va"]))

        return per_edge_outliers

    def train_isolation_forest_PMU_caseA(self, X_train):

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

    def predict_outliers_PMU_caseA(self, X_test, outlier_dict):
        per_edge_outliers = {
            "Im":  [],
            "Vm":  [],
            "Va":  [],
            "ALL": [],
        }

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        # TODO ALL
        # Keep only meter relevant information and predict per meter
        dataset = X_test[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]
        predictions = self.model_ALL.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = outlier_dict["ALL"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR ALL MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%", "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["ALL"] = list(predicted_outlier_indices)

        # TODO Im
        dataset = X_test[:, [2 * NUM_NODES + edge_idx]]
        predictions = self.model_Im.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = outlier_dict["Im"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Im MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%", "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Im"] = list(predicted_outlier_indices)

        # TODO Vm
        dataset = X_test[:, [node_idx]]
        predictions = self.model_Vm.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = outlier_dict["Vm"]

        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Vm MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%", "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Vm"] = list(predicted_outlier_indices)

        # TODO Va
        # Keep only meter relevant information and predict per meter
        dataset = X_test[:, [NUM_NODES + node_idx]]
        predictions = self.model_Va.predict(dataset)
        predicted_outlier_indices = list(np.where(predictions == -1)[0])
        real_outliers = outlier_dict["Va"]
        set_diff = set(real_outliers) - set(predicted_outlier_indices)
        print("NOT FOUND FOR Va MEASUREMENTS ERRORS: ", 100 * (len(set_diff) / len(real_outliers)), "%", "Real Outliers: ", len(real_outliers), "Predicted Outliers:", len(predicted_outlier_indices))
        per_edge_outliers["Va"] = list(predicted_outlier_indices)

        per_edge_outliers = self.divide_outlier_indices(per_edge_outliers)

        return per_edge_outliers

class KNNImputerMeasurements:

    def __init__(self, meter):
        self.KNNImputer         = KNNImputer(n_neighbors=10)
        self.meter              = meter

    def train_KNNImputers_PMU_caseA(self, X_train, X_train_outlier_index_dict):

        outlier_indices = list(set(X_train_outlier_index_dict["Vm"] +
                                   X_train_outlier_index_dict["Va"] +
                                   X_train_outlier_index_dict["Im"] +
                                   X_train_outlier_index_dict["ALL"]))

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        inliners = X_train[~np.isin(np.arange(len(X_train)), outlier_indices)]
        print(inliners.shape)

        inliners = inliners[:, [node_idx, NUM_NODES+node_idx, 2*NUM_NODES+edge_idx, 2*NUM_NODES+NUM_BRANCHES+edge_idx]]
        self.KNNImputer.fit(inliners)


    def impute_values_PMU_caseA(self, X_test, X_test_outlier_index_dict, compare_np):

        edge_idx = self.meter
        node_idx = branch_data[self.meter]["sending_node"]

        # TODO Get indices of meter only
        X_train    = X_test[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]
        compare_np = compare_np[:, [node_idx, NUM_NODES + node_idx, 2 * NUM_NODES + edge_idx, 2 * NUM_NODES + NUM_BRANCHES + edge_idx]]

        # TODO Vm
        outlier_indices = X_test_outlier_index_dict["Vm"]
        X_train[outlier_indices, 0] = np.nan
        X_train_Vm = self.KNNImputer.transform(X_train)
        original_Vm = compare_np[outlier_indices, 0].flatten().tolist()
        imputed_Vm  = X_train_Vm[outlier_indices, 0].flatten().tolist()
        mape = sum([abs(original_Vm[i] - imputed_Vm[i]) / original_Vm[i] for i in range(len(original_Vm))])/len(original_Vm)
        print("MAPE Imputed Vm: ", mape, " - Imputed values: ", len(imputed_Vm))

        # TODO Va
        outlier_indices = X_test_outlier_index_dict["Va"]
        X_train[outlier_indices, 1] = np.nan
        X_train_Va = self.KNNImputer.transform(X_train)
        original_Va = compare_np[outlier_indices, 1].flatten().tolist()
        imputed_Va = X_train_Va[outlier_indices, 1].flatten().tolist()
        mae = sum([abs(original_Va[i] - imputed_Va[i]) for i in range(len(original_Va))]) / len(original_Va)
        print("MAE Imputed Va: ", mae, " - Imputed values: ", len(imputed_Va))

        # TODO Im
        outlier_indices = X_test_outlier_index_dict["Im"]
        X_train[outlier_indices, 2] = np.nan
        X_train_Im = self.KNNImputer.transform(X_train)
        original_Im = compare_np[outlier_indices, 2].flatten().tolist()
        imputed_Im = X_train_Im[outlier_indices, 2].flatten().tolist()
        mape = sum([abs(original_Im[i] - imputed_Im[i]) / original_Im[i] for i in range(len(original_Im))]) / len(original_Im)
        print("MAPE Imputed Im: ", mape, " - Imputed values: ", len(imputed_Im))

        # Impute values on original frame X_dataset
        Im_outlier_indices = X_test_outlier_index_dict["Im"]
        Vm_outlier_indices = X_test_outlier_index_dict["Vm"]
        Va_outlier_indices = X_test_outlier_index_dict["Va"]

        X_test[Im_outlier_indices, 2*NUM_NODES + edge_idx]   = imputed_Im
        X_test[Vm_outlier_indices, node_idx]                 = imputed_Vm
        X_test[Va_outlier_indices, NUM_NODES + node_idx]     = imputed_Va

        return X_test

if __name__ == "__main__":

    meter = 75

    # TODO Conventional
    X_train = np.load("datasets/95UKGDPMU_caseA_input.npy")
    y_train  = np.load("datasets/95UKGDPMU_caseA_output.npy")

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    X_old   = X_test.copy()

    #TODO - Outlier injection into the datasets X_train, X_test
    out_inj_X_train = OutlierInjector(X_train)
    X_train_outlier_arr, X_train_outlier_index_dict = out_inj_X_train.inject_outliers_PMUcaseA([meter], prob_m=0.01, prob_a=0.01)
    out_inj_X_test = OutlierInjector(X_test)
    X_test_outlier, X_test_outlier_index_dict = out_inj_X_test.inject_outliers_PMUcaseA([meter], prob_m=0.01, prob_a=0.01)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    print("X_test outlier shape: ", X_test_outlier.shape)
    #TODO - Isolation Forest Outlier Detection - Gets outlier X_train dataset as input
    iso_detector = IF_OutlierDetection(meter)
    iso_detector.train_isolation_forest_PMU_caseA(X_train_outlier_arr)
    per_edge_outliers = iso_detector.predict_outliers_PMU_caseA(X_test_outlier, X_test_outlier_index_dict)


    #TODO - KNN imputer to impute values
    knn_imp = KNNImputerMeasurements(meter)
    knn_imp.train_KNNImputers_PMU_caseA(X_train_outlier_arr, X_train_outlier_index_dict)
    X_test_imputed = knn_imp.impute_values_PMU_caseA(X_test_outlier, X_test_outlier_index_dict,compare_np=X_old)

    #TODO - From X_imputed and t_test, we need to remove the indices of all PMU measurements wrong
    X_test_imputed = X_test_imputed[~np.isin(np.arange(len(X_test_imputed)), per_edge_outliers["ALL"])]
    y_test_imputed = y_test[~np.isin(np.arange(len(y_test)), per_edge_outliers["ALL"])]
    print("X_test imputed shape: ", X_test_imputed.shape)
    print("y_test imputed shape: ", y_test_imputed.shape)
    #TODO Here add y_test ALL_outlier indices exclusion

