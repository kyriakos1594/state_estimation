from state_estimation import *
from topology_identification import *
from datetime import datetime, timedelta

class SMD_placement:

    def __init__(self, model, PP, subPP):

        self.model = model
        self.PP = PP
        self.subPP    = subPP
    def execute_NN(self):

        FS = self.PP
        method = self.subPP
        model = self.model
        TI_acc_threshold = 0.95

        PreProc_TI = Preprocess()
        X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, y_test_labels = PreProc_TI.preprocess_data()

        start_TI = datetime.now()

        print(f"executing SMD placement utilizing NN with feature selection {self.PP} and {self.subPP}")

        TI_Estimator = TIPredictorTrainProcess(TI_acc_threshold, model, X_train, y_train_labels, X_val, y_val_labels, X_test, y_test_labels, FS=FS, method=method)
        TI_PMUs = TI_Estimator.execute_NN()

        end_TI = start_SE = datetime.now()
        delta_TI = end_TI - start_TI

        SE_Estimator = DSSE_Estimator_TrainProcess(model, X_train, y_train_outputs, X_val, y_val_outputs, X_test, y_test_outputs, old_PMUs=TI_PMUs, FS=FS, method=method)
        used_features = SE_Estimator.execute()

        end_SE = datetime.now()

        delta_SE = end_SE - start_SE

        print("TI TIME: ", delta_TI, "SE TIME", delta_SE)


    def execute_GNN(self):

        model = self.model
        TI_acc_th = 0.95

        PreProc_TI = Preprocess()
        X_train, y_train_outputs, y_train_labels, X_val, y_val_outputs, y_val_labels, X_test, y_test_outputs, y_test_labels = PreProc_TI.preprocess_data()

        TI_PTP = TIPredictorTrainProcess(TI_acc_th, model, X_train, y_train_labels, X_val, y_val_labels, X_test, y_test_labels, self.PP, self.subPP)
        TI_PMUs = TI_PTP.execute_GNN()

        print(f"executing SMD placement utilizing GNN with feature selection {self.PP} and {self.subPP}")

        DSSE_FS = DSSE_Estimator_TrainProcess(model=model,
                                              X_train=X_train,
                                              y_train=y_train_outputs,
                                              X_val=X_val,
                                              y_val=y_val_outputs,
                                              X_test=X_test,
                                              y_test=y_test_outputs,
                                              old_PMUs=TI_PMUs,
                                              FS=self.PP,
                                              method=self.subPP)
        DSSE_PMUs = DSSE_FS.execute_GNN()

        print("Required SMDs: ", DSSE_PMUs)

    def execute(self):

        if self.model == "NN":
            self.execute_NN()
        elif self.model == "GNN":
            self.execute_GNN()
        else:
            print("Unidentified model")


if __name__ == "__main__":

    SMDPlacement = SMD_placement("NN", "RF", "rfe")
    #SMDPlacement.execute_GNN()
    SMDPlacement.execute()

    #TODO Implement most usefule features
    #from sklearn.feature_selection import RFE
    #from sklearn.ensemble import RandomForestClassifier

    #model = RandomForestClassifier()
    #rfe = RFE(model, n_features_to_select=k)
    #rfe.fit(X, y)
    #selected_features = rfe.support_
