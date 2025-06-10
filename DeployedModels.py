import keras
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymssql
import json


class DeployedModels:

    def __init__(self):

        self.scaler    = None
        self.TI_model  = None
        self.SEm_model = None
        self.SEa_model = None
        self.NUM_NODES = 131
        self.used_indices = [124, 127, 128, 255, 258, 259, 386, 389, 390, 517, 520, 521]

        self.conn = None
        self.cur  = None

    def db_conn(self):
        self.conn = pymssql.connect(server='147.102.30.47',
                               user='opentunity',
                               password='0pentunity44$$',
                               database='opentunity_dev')

        self.cursor = self.conn.cursor()

    def db_disconn(self):

        self.conn.close()
        self.cursor.close()

    def get_data(self, ts_str):

        sql = f"""select m.Timestamp, a.AssetName, mt.Name, m.Value  
                from 			Measurements 	  m 
                inner join 		Assets 			  a ON  a.AssetID=m.AssetID
                inner join 		MeasurementTypes mt ON mt.MeasurementID=m.MeasurementID
                    where m.[Timestamp] = '__ts_str__'
                		and   a.AssetID IN (13, 14, 15)
		                order by mt.Name DESC, a.AssetName;"""
        sql = sql.replace("__ts_str__", ts_str)
        self.cursor.execute(sql)


        df_measurements = pd.DataFrame(self.cursor)
        print(df_measurements)
        df_measurements.columns = ["Timestamp", "AssetName", "Measurement_Type", "Value"]

        # Vm Values
        Vm_124 = df_measurements[(df_measurements["AssetName"]=="22") & (df_measurements["Measurement_Type"]=="Voltage")]["Value"].values.tolist()[0]
        Vm_127 = df_measurements[(df_measurements["AssetName"]=="44") & (df_measurements["Measurement_Type"]=="Voltage")]["Value"].values.tolist()[0]
        Vm_128 = df_measurements[(df_measurements["AssetName"]=="48") & (df_measurements["Measurement_Type"]=="Voltage")]["Value"].values.tolist()[0]

        # Va Values
        Va_124 = df_measurements[(df_measurements["AssetName"]=="22") & (df_measurements["Measurement_Type"]=="Voltage_Angle")]["Value"].values.tolist()[0]
        Va_127 = df_measurements[(df_measurements["AssetName"]=="44") & (df_measurements["Measurement_Type"]=="Voltage_Angle")]["Value"].values.tolist()[0]
        Va_128 = df_measurements[(df_measurements["AssetName"]=="48") & (df_measurements["Measurement_Type"]=="Voltage_Angle")]["Value"].values.tolist()[0]

        # Im_inj Values
        Im_124 = df_measurements[(df_measurements["AssetName"]=="22") & (df_measurements["Measurement_Type"]=="Current")]["Value"].values.tolist()[0]
        Im_127 = df_measurements[(df_measurements["AssetName"]=="44") & (df_measurements["Measurement_Type"]=="Current")]["Value"].values.tolist()[0]
        Im_128 = df_measurements[(df_measurements["AssetName"]=="48") & (df_measurements["Measurement_Type"]=="Current")]["Value"].values.tolist()[0]

        # Ia_inj Values
        Ia_124 = df_measurements[(df_measurements["AssetName"]=="22") & (df_measurements["Measurement_Type"]=="Current_Angle")]["Value"].values.tolist()[0]
        Ia_127 = df_measurements[(df_measurements["AssetName"]=="44") & (df_measurements["Measurement_Type"]=="Current_Angle")]["Value"].values.tolist()[0]
        Ia_128 = df_measurements[(df_measurements["AssetName"]=="48") & (df_measurements["Measurement_Type"]=="Current_Angle")]["Value"].values.tolist()[0]

        measurements = [Vm_124, Vm_127, Vm_128, Va_124, Va_127, Va_128, Im_124, Im_127, Im_128, Ia_124, Ia_127, Ia_128]
        measurements = np.array(measurements)
        measurements = measurements.reshape(1, -1)

        return measurements

    def load_scaler(self):
        self.scaler = joblib.load('scalers/MESOGEIA_NN_PMUB_METERS_StandardScaler.pkl')

    def load_models(self):

        self.TI_model  = keras.models.load_model("DeployedModels/MESOGEIA_NN_METERS_127_128_124_TI.h5")
        self.SEm_model = keras.models.load_model("DeployedModels/MESOGEIA_NN_NETERS_MAGNITUDES_127_128_124_SE.h5")
        self.SEa_model = keras.models.load_model("DeployedModels/MESOGEIA_NN_NETERS_ANGLES_127_128_124_SE.h5")

    def return_used_indices(self):
        used_feature_indices = []
        for meter_index in [127, 128, 124]:
            used_feature_indices = used_feature_indices + [meter_index,
                                                   self.NUM_NODES + meter_index,
                                                   2 * self.NUM_NODES + meter_index,
                                                   3 * self.NUM_NODES + meter_index]

        return used_feature_indices

    def mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def test_output(self):

        df = pd.read_csv("datasets/MESOGEIA.csv")
        input_df = df[(df["TopNo"]==1) & (df["Simulation"]==7301)]

        vm          = input_df["vm_pu"].values.tolist()[:-2]
        va          = input_df["va_degree"].values.tolist()[:-2]
        Im_inj      = input_df["Im_inj"].values.tolist()[:-2]
        Ia_inj      = input_df["Ia_inj"].values.tolist()[:-2]

        used_indices = sorted([128, 258, 259, 255, 389, 390, 386, 520, 521, 517, 124, 127])
        print(used_indices)

        l = np.array(vm + va + Im_inj + Ia_inj)

        print("Unscaled: ", l.shape)
        l = l[used_indices]

        l = l.reshape(1, -1)

        l = self.scaler.transform(l)

        print("Scaled: ", l)
        TI = self.TI_model.predict(l)
        print(used_indices)
        print(TI)
        self.SEm_model = keras.models.load_model("DeployedModels/MESOGEIA_NN_NETERS_MAGNITUDES_127_128_124_SE.h5")
        Vm = self.SEm_model.predict(l)
        vm = np.array(vm)
        print(Vm)
        print(vm)
        print(self.mape(vm, Vm), "%")
        #Va = self.SEa_model.predict(l)
        #print(Va)

    def predict_timestamp(self, timestamp_str):

        measurements = self.get_data(timestamp_str)
        measurements = self.scaler.transform(measurements)
        message_dict = {}

        print("Scaled: ", measurements)
        TI = self.TI_model.predict(measurements)
        max_index = np.argmax(TI)
        topology = max_index + 1

        Vm = self.SEm_model.predict(measurements)
        #print(Vm)

        Va = self.SEa_model.predict(measurements)
        #print(Va)

        TI = "T" + str(topology)
        print(TI)
        Vm = [round(v, 4) for v in Vm.tolist()[0]]
        Va = [round(v, 4) for v in Va.tolist()[0]]

        message_dict["Timestamp"] = timestamp_str
        message_dict["TI"] = TI
        message_dict["buses"] = {}

        data = [{"bus_id": i, "voltage_pu": Vm[i], "angle_deg": Va[i]} for i in range(len(Vm))]

        message_dict = {
            "Timestamp": timestamp_str,
            "TI": TI,
            "buses": data
        }

        return json.dumps(message_dict)

    def execute_prediction(self):

        self.db_conn()
        self.load_scaler()
        self.load_models()
        msg = self.predict_timestamp('2022-12-03 00:00:00.000')
        self.db_disconn()

        return msg


if __name__=="__main__":
    DepMod = DeployedModels()
    res = DepMod.execute_prediction()
    print(res)