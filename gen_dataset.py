import pandapower as pp
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from IEEE_datasets.IEEE33 import config_dict
from config_file import *
from pandapower.pypower.makeYbus import makeYbus
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt

class AllignDataProfiles:

    def __init__(self):
        self.basefolder         = "/opt/emie/pythonProject/state_estimation/datasets/"
        self.wind_folder        = self.basefolder + "WIND_data/"
        self.MV_demand_folder   = self.basefolder + "MV_DEMAND_data/"
        self.solar_folder       = self.basefolder + "PV_data/"
        self.alligned_solar     = self.basefolder + "PV_alligned.csv"
        self.alligned_wind      = self.basefolder + "WD_alligned.csv"
        self.alligned_SCADA     = self.basefolder + "MV_DEMAND_SCADA_alligned.csv"
        self.alligned_X84       = self.basefolder + "MV_DEMAND_X84_alligned.csv"
        self.alligned_LV        = self.basefolder + "LV_alligned.csv"
        self.alligned_profiles  = self.basefolder + "alligned_profiles.csv"
        self.scaled_alligned_profiles = self.basefolder + "scaled_alligned_profiles.csv"
        self.weekly_alligned_scaled_profiles = self.basefolder + "weekly_alligned_scaled_profiles.csv"

    def read_solar_timestamps(self, ts):

        ts = datetime.strptime(ts, '%d/%m/%Y %H:%M')
        return ts.strftime("%Y-%m-%d %H:%M:00")
    def read_wind_timestamps(self, ts):
        ts = datetime.strptime(ts, '%d %m %Y %H:%M')
        return ts.strftime("%Y-%m-%d %H:%M:00")
    def read_X84_timestamps(self, ts):
        ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        return ts.strftime("%Y-%m-%d %H:%M:00")
    def alter_solar_record(self, ts):

        date_str, hour_str = ts.split(" ")
        year, month, day = date_str.split("-")

        if (year == "2019") and (int(month) >= 3):
            return "2021-"+month+"-"+day+" "+hour_str
        elif (year == "2020") and (int(month) <= 3):
            return "2021-"+month+"-"+day+" "+hour_str
    def read_lv_timestamps(self, ts):
        return ts.replace("2016", "2021").replace("T", " ").replace("Z", "")
    def process_PV_data(self):

        df_list = []

        for root, dirs, files in os.walk(self.solar_folder, topdown=False):
            for file in files:
                df = pd.read_csv(self.solar_folder+file, delimiter="\t", encoding="utf16")
                df = df[["Timestamp", "Value_KWh"]]
                df["Timestamp"] = df["Timestamp"].apply(lambda x: self.read_solar_timestamps(x))
                header = "PV_profile " + file.split(".")[0][-1]
                df.columns = ["Timestamp", header]
                df_list.append(df)

        target_df = df_list[0]
        for df in df_list[1:]:
            target_df = pd.merge(target_df, df, how='inner', on='Timestamp')
        target_df = target_df.sort_values(by="Timestamp", ascending=True)

        target_df.to_csv(self.alligned_solar)

        return self.alligned_solar
    def process_WD_data(self):

        for root, dirs, files in os.walk(self.wind_folder, topdown=False):
            for file in files:
                df = pd.read_csv(self.wind_folder+file)
                df = df[["Date/Time", "LV ActivePower (kW)"]]
                df.columns = ["Timestamp", "WD 1"]
                df["Timestamp"] = df["Timestamp"].apply(lambda x: self.read_wind_timestamps(x))
                df = df.sort_values(by="Timestamp", ascending=True)

        # Make data for 2020-2021
        df_2020 = df
        df_2020["Timestamp"] = df_2020["Timestamp"].apply(lambda x: "2020" + x[4:])
        df_2021 = df
        df_2021["Timestamp"] = df_2021["Timestamp"].apply(lambda x: "2021" + x[4:])

        df.to_csv(self.alligned_wind)
    def preprocess_MV_DEMAND_data_SCADA(self):

        df_list = []

        for root, dir, files in os.walk(self.MV_demand_folder):
            for file in files:
                if ("R210" in file) and ("loc" not in file):
                    gen_df = pd.DataFrame()
                    df = pd.read_excel(self.MV_demand_folder+file)
                    df = df.loc[1:, :]
                    df.columns = ["Χρόνος SCADA","Χρόνος πηγής","Ετικέτα","Περιγραφή","Τιμή","Μονάδα"]
                    df = df[["Χρόνος πηγής", "Τιμή"]]
                    df.columns = ["Timestamp", "R210 Value"]
                    df["DateHour"] = df["Timestamp"].apply(lambda x: x.split(":")[0])
                    df = df.sort_values(by="Timestamp", ascending=True)
                    # Get date list
                    datehour_list =list(set([x.split(":")[0] for x in df["Timestamp"].values.tolist()]))
                    for datehour in datehour_list:
                        tmp_list = []
                        tmp_df = df[df["DateHour"] == datehour].reset_index()
                        #print(datehour)
                        N = tmp_df.shape[0]

                        # DT 00:00:00
                        index00 = 0
                        rec00_value = tmp_df.loc[index00, :].values.tolist()[2]
                        tmp_list.append([datehour+":00:00", rec00_value])

                        # DT 15:00:00
                        index15 = N//4
                        rec15_value = tmp_df.loc[index15, :].values.tolist()[2]
                        tmp_list.append([datehour+":15:00", rec15_value])

                        # DT 30:00:00
                        index30 = 2 * (N//4)
                        rec30_value = tmp_df.loc[index30, :].values.tolist()[2]
                        tmp_list.append([datehour+":30:00", rec30_value])

                        # DT 45:00:00
                        index45 = 3 * (N//4)
                        rec45_value = tmp_df.loc[index45, :].values.tolist()[2]
                        tmp_list.append([datehour+":45:00", rec45_value])

                        merge_df = pd.DataFrame(tmp_list, columns=["Timestamp", "R210 Load"])
                        df_list.append(merge_df)

        R210_df = df_list[0]

        for df in df_list[1:]:
            R210_df = pd.concat([R210_df, df], ignore_index=True)

        R210_df = R210_df[["Timestamp", "R210 Load"]].sort_values(by="Timestamp", ascending=True)
        R210_df.to_csv(self.alligned_SCADA)

        return self.alligned_SCADA
    def preprocess_MV_DEMAND_data_X84(self):

        df_list = []
        res_df = pd.DataFrame()
        for root, dirs, files in os.walk(self.MV_demand_folder):
            for file in files:
                i=0
                if ("X84" in file) and ("loc" not in file):
                    i+=1
                    df = pd.read_excel(self.MV_demand_folder+file)
                    df = df[["Quarter_time_stamp", "Value"]]
                    df.columns = ["Timestamp", f"X84_{str(i)} MV Load"]
                    df["Timestamp"] = df["Timestamp"].apply(lambda x: str(x))
                    df["Timestamp"] = df["Timestamp"].apply(lambda x: self.read_X84_timestamps(x))
                    df = df.sort_values(by="Timestamp", ascending=True)
                    df_list.append(df)

        target_df = pd.merge(df_list[0], df_list[1], how="inner", on="Timestamp")
        target_df.to_csv(self.alligned_X84)

        return self.alligned_X84
    def preprocess_LV_DEMAND_data(self):

        df = pd.read_csv(self.basefolder+"LV_DEMAND_data/residential_profiles.csv").dropna(subset=["utc_timestamp"])
        df = df[["utc_timestamp", "Residential 1", "Residential 2", "Residential 3", "Residential 4",
                 "Residential 5", "Residential 6"]]
        df.columns = ["Timestamp", "R1", "R2", "R3", "R4", "R5", "R6"]
        df["Timestamp"] = df["Timestamp"].apply(lambda x: str(x))
        df["Timestamp"] = df["Timestamp"].apply(lambda x: self.read_lv_timestamps(x))
        df = df.dropna(subset=["Timestamp"])

        df.to_csv(self.alligned_LV)

        return self.alligned_LV
    def change_resolution_from_10_to_15(self, df_wind):

        df_list = []

        df_wind = df_wind.sort_values(by="Timestamp", ascending=True)
        df_wind["DateHour"] = df_wind["Timestamp"].apply(lambda x: x.split(":")[0])
        datehour_list = list(set([x.split(":")[0] for x in df_wind["Timestamp"].values.tolist()]))

        for datehour in datehour_list:
            tmp_list = []
            source_df = df_wind[df_wind["DateHour"]==datehour][["Timestamp", "WD 1"]]

            # Index 00:00
            record00 = source_df.iloc[0, :].values.tolist()
            record00 = [datehour+":00:00", record00[1]]

            # Index 15:00
            if source_df.shape[0] < 2:
                record15 = source_df.iloc[0, :]
            else:
                record15 = source_df.iloc[1, :]
            record15 = [datehour + ":15:00", record15[1]]

            # Index 30:00
            if source_df.shape[0] < 3:
                record30 = source_df.iloc[0, :]
            else:
                record30 = source_df.iloc[2, :]
            record30 = [datehour + ":30:00", record30[1]]


            # Index 45:00
            if source_df.shape[0] < 4:
                record45 = source_df.iloc[0, :]
            else:
                record45 = source_df.iloc[3, :]
            record45 = [datehour + ":45:00", record45[1]]


            df_list.append(pd.DataFrame([record00, record15, record30, record45], columns=["Timestamp", "WD 1"]))

        result_df = df_list[0]

        for df in df_list[1:]:
            result_df = pd.concat([result_df, df], ignore_index=True)

        result_df = result_df.sort_values(by="Timestamp", ascending=True)

        return result_df
    def allign_datasets(self):

        df_solar = pd.read_csv(self.alligned_solar).iloc[1:, :].drop(columns=['Unnamed: 0'], errors='ignore')
        df_wind  = pd.read_csv(self.alligned_wind).iloc[1:, :].drop(columns=['Unnamed: 0'], errors='ignore')
        df_X84   = pd.read_csv(self.alligned_X84).iloc[1:, :].drop(columns=['Unnamed: 0'], errors='ignore')
        df_SCADA = pd.read_csv(self.alligned_SCADA).iloc[1:, :].drop(columns=['Unnamed: 0'], errors='ignore')
        df_LV    = pd.read_csv(self.alligned_LV).dropna(subset=["Timestamp"]).iloc[1:, :].drop(columns=['Unnamed: 0'], errors='ignore')

        print("BEFORE ALLIGNMENT - SOLAR: ", df_solar["Timestamp"].min(), df_solar["Timestamp"].max())
        print("BEFORE ALLIGNMENT - WIND: ", df_wind["Timestamp"].min(), df_wind["Timestamp"].max())
        print("BEFORE ALLIGNMENT - X84: ", df_X84["Timestamp"].min(), df_X84["Timestamp"].max())
        print("BEFORE ALLIGNMENT - SCADA: ", df_SCADA["Timestamp"].min(), df_SCADA["Timestamp"].max())
        print("BEFORE ALLIGNMENT - LV: ", df_LV["Timestamp"].min(), df_LV["Timestamp"].max())


        # SOLAR:  2019-02-27 13:30:00 - 2020-05-06 00:00:00 -> turn 2019 into 2021 and keep 2021
        # WIND:   2021-01-01 00:00:00 - 2021-12-31 23:50:00 -> make no changes
        # X84:    2020-03-29 03:15:00 - 2022-08-01 00:00:00 -> keep only 2021
        # SCADA:  2020-01-01 00:00:00 - 2021-12-31 23:45:00 -> keep only 2021

        # TODO Solar - Alter year 2019 and 2020 into 2021 and keep 2021 only
        df_solar["Timestamp"] = df_solar["Timestamp"].apply(lambda x: self.alter_solar_record(x))
        df_solar = df_solar[df_solar["Timestamp"].str.contains("2021-", case=False, na=False)]
        print("SOLAR: ", df_solar["Timestamp"].min(), df_solar["Timestamp"].max(), df_solar.shape[0])

        #TODO Wind - Keep year as is but keep 1st, 2nd, 3rd and 4th records as quarters
        df_wind = self.change_resolution_from_10_to_15(df_wind)
        print("WIND: ", df_wind["Timestamp"].min(), df_wind["Timestamp"].max(), df_wind.shape[0])

        #TODO Demand SCADA - Keep only year 2021
        df_SCADA = df_SCADA[df_SCADA["Timestamp"].str.contains("2021-", case=False, na=False)]
        print("SCADA: ", df_SCADA["Timestamp"].min(), df_SCADA["Timestamp"].max(), df_SCADA.shape[0])

        #TODO Demand X84 - Keep only year 2021
        df_X84 = df_X84[df_X84["Timestamp"].str.contains("2021-", case=False, na=False)]
        print("X84: ", df_X84["Timestamp"].min(), df_X84["Timestamp"].max(), df_X84.shape[0])

        #TODO Demand LV - Keep only year 2021
        print("LV: ", df_LV["Timestamp"].min(), df_LV["Timestamp"].max(), df_LV.shape[0])


        #SOLAR:  2021-01-01 00:00:00 - 2021-12-31 23:45:00
        #WIND:   2021-01-01 00:00:00 - 2021-12-31 23:45:00
        #SCADA:  2021-01-01 00:00:00 - 2021-12-31 23:45:00
        #X84:    2021-01-01 00:00:00 - 2021-11-01 14:30:00

        initial_df = df_solar

        for df in [df_wind, df_SCADA, df_X84, df_LV]:
            initial_df = pd.merge(initial_df, df, how="inner", on="Timestamp", suffixes=('_left', '_right'))

        initial_df = initial_df.drop_duplicates(subset=["Timestamp"], keep="last")
        initial_df.sort_values(by="Timestamp", ascending=True)
        initial_df.to_csv(self.alligned_profiles)

        #TODO Scale values of columns

    def scale_profiles(self):

        df = pd.read_csv(self.alligned_profiles)
        target_columns = ["PV_profile 2", "PV_profile 5", "PV_profile 6", "PV_profile 3", "PV_profile 4",
                          "PV_profile 7", "PV_profile 1", "WD 1", "R210 Load", "X84_1 MV Load_x", "X84_1 MV Load_y",
                          "R1", "R2", "R3", "R4", "R5", "R6"]

        for column in target_columns:
            df[column] = df[column].apply(lambda x: str(x))
            df[column] = df[column].apply(lambda x: float(x.replace(",", ".")))
            max_value  = df[column].max()
            min_value  = df[column].min()
            df[column] = df[column].apply(lambda x: (x - min_value)/(max_value - min_value))

        df = df[["Timestamp"]+target_columns]

        df.to_csv(self.scaled_alligned_profiles)
    def sample_weekly_data(self):

        df = pd.read_csv(self.scaled_alligned_profiles)
        df["Timestamp"] = df["Timestamp"].apply(lambda x:str(x))

        df_list = []

        #TODO Get only first week out of all data
        for month in range(1, 12+1):
            month = str(month) if month >= 10 else "0"+str(month)
            for day in range(1, 7+1):
                day = str(day) if day >= 10 else "0" + str(day)
                date_str = "2021"+"-"+str(month)+"-"+str(day)
                tmp_df = df[df["Timestamp"].str.contains(date_str)]
                df_list.append(tmp_df)

        final_df = df_list[0]

        for df in df_list[1:]:
            final_df = pd.concat([final_df, df], ignore_index=True)

        final_df = final_df.sort_values(by="Timestamp", ascending=True)
        final_df.to_csv(self.weekly_alligned_scaled_profiles)

class LoadPowerFlow:

    def __init__(self, filename, dataset, topology, datetime_str, profile_dict):

        self.filename = filename
        self.filepath = "datasets/"+self.filename
        self.net      = None
        self.Sbase    = None
        self.Vbase    = None
        self.dataset  = dataset
        self.topology = topology
        self.datetime_str = datetime_str
        self.profile_dict = profile_dict

    def change_topology(self, net):
        if self.dataset == "IEEE33":
            open_branches = config_dict[self.dataset][self.topology]["open_branches"]
            net.line["in_service"] = pd.Series(False if (i in open_branches) else True for i in range(net.line.shape[0]))
        elif self.dataset == "MESOGEIA":
            open_branches = config_dict[self.dataset][self.topology]["open_branches"]
            net.line["in_service"] = pd.Series(False if (i in open_branches) else True for i in range(net.line.shape[0]))
        elif self.dataset == "95UKGD":
            open_branches = config_dict[self.dataset][self.topology]["open_branches"]
            net.line["in_service"] = pd.Series(False if (i in open_branches) else True for i in range(net.line.shape[0]))

    #TODO Now they change all the time
    def randomize_loads(self, net):
        self.net.load["LOAD_CHANGE_FLAG"] = np.random.choice([0, 1], size=len(self.net.load))
        #self.net.load["LOAD_CHANGE_FLAG"] = np.array([1 for i in range(self.net.load.shape[0])])
        #TODO 1 always
        r = 0.30
        # var_vector (same size as the number of loads) defines the scaling factors
        # For simplicity, we'll use a vector of ones. Adjust as needed.
        var_vector = np.ones(len(net.load))
        # Apply randomization to active power (p_mw)
        # We now apply the changes to each load based on the load change flag
        #print(net.load["LOAD_CHANGE_FLAG"])
        net.load["p_mw"] = net.load.apply(
            lambda row: row["p_mw"] * (1 + row["LOAD_CHANGE_FLAG"] * (-r + (2 * r) * np.random.rand())),
            axis=1
        )

        # Υπολογισμός power factor από τις υπάρχουσες τιμές P και Q
        pf = net.load["p_mw"] / np.sqrt(net.load["p_mw"] ** 2 + net.load["q_mvar"] ** 2)
        # Apply randomization to reactive power (q_mvar)
        net.load["q_mvar"] = np.tan(np.arccos(pf)) * net.load["p_mw"]

    def insert_noise(self, df_V, df_I):

        # Define constants
        me_phasor_mag = 0.001  # pu
        me_phasor_ang = 0.018  # degrees

        # Keep original
        df_V['vm_pu_actual']     = df_V['vm_pu'].copy()
        df_V['va_degree_actual'] = df_V['va_degree'].copy()


        df_V['vm_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V['va_degree'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_V)))

        df_V['P_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V.loc[0, 'P_pu'] = 0

        df_V['Q_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V.loc[0, 'Q_pu'] = 0

        df_V['Im_inj'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V['Ia_inj'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_V)))

        # Insert noise for branches
        # Add measurement noise
        # Add noisy magnitude (Im_pu) and angle (Ia_pu in degrees)
        df_I['Im_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_I)))
        df_I['Ia_pu'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_I)))

        return df_V, df_I

    def assign_buses(self, net, dataset):

        df_profiles = pd.read_csv(PROFILES_FILEPATH)
        df_profiles = df_profiles[df_profiles["Timestamp"] == self.datetime_str]

        bus_config_dict = bus_types
        # Iterate over the buses and print index along with bus details
        for index, row in net.bus.iterrows():

            # Slack bus
            if index in bus_types["slack"]:
                net.bus.loc[index, ["type", "name"]] = ["ref", f"Slack Bus {str(index)}"]

            elif index in bus_types["PV_wind"]:
                net.bus.loc[index, ["type", "name"]] = ["b", f"PV bus {str(index)} - WIND - {self.profile_dict[index]}"]
                if index in net.sgen["bus"].values.tolist():
                    P_gen   = net.sgen[net.sgen["bus"] == index]["p_mw"].values.tolist()[0]
                    profile = df_profiles[self.profile_dict[index]]
                    P_bus = P_gen * profile
                    net.sgen[net.sgen["bus"] == index]["p_mw"]   = P_bus
                    net.sgen[net.sgen["bus"] == index]["vm_pu"]  = 1
                    net.sgen[net.sgen["bus"] == index]["q_mvar"] = 0.0  # Set Q to 0 (variable)
                    net.sgen[net.sgen["bus"] == index]["controllable"] = True  # Make them controllable
                elif index in net.gen["bus"].values.tolist():
                    P_gen = net.gen[net.gen["bus"] == index]["p_mw"].values.tolist()[0]
                    profile = df_profiles[self.profile_dict[index]]
                    P_bus = P_gen * profile
                    net.gen[net.gen["bus"] == index]["p_mw"] = P_bus
                    net.gen[net.gen["bus"] == index]["vm_pu"] = 1
                    net.gen[net.gen["bus"] == index]["q_mvar"] = 0.0  # Set Q to 0 (variable)
                    net.gen[net.gen["bus"] == index]["controllable"] = True  # Make them controllable

            elif index in bus_types["PV_solar"]:
                net.bus.loc[index, ["type", "name"]] = ["b", f"PV bus {str(index)} - SOLAR - {self.profile_dict[index]}"]
                if index in net.sgen["bus"].values.tolist():
                    P_gen   = net.sgen[net.sgen["bus"] == index]["p_mw"].values.tolist()[0]
                    profile = df_profiles[self.profile_dict[index]]
                    P_bus = P_gen * profile
                    net.sgen[net.sgen["bus"] == index]["p_mw"]   = P_bus
                    net.sgen[net.sgen["bus"] == index]["vm_pu"]  = 1
                    net.sgen[net.sgen["bus"] == index]["q_mvar"] = 0.0  # Set Q to 0 (variable)
                    net.sgen[net.sgen["bus"] == index]["controllable"] = True  # Make them controllable
                elif index in net.gen["bus"].values.tolist():
                    P_gen = net.gen[net.gen["bus"] == index]["p_mw"].values.tolist()[0]
                    profile = df_profiles[self.profile_dict[index]]
                    P_bus = P_gen * profile
                    net.gen[net.gen["bus"] == index]["p_mw"] = P_bus
                    net.gen[net.gen["bus"] == index]["vm_pu"] = 1
                    net.gen[net.gen["bus"] == index]["q_mvar"] = 0.0  # Set Q to 0 (variable)
                    net.gen[net.gen["bus"] == index]["controllable"] = True  # Make them controllable

            elif index in bus_types["PQ_MV"]:
                net.bus.loc[index, ["type", "name"]] = ["b", f"PQ bus {str(index)} - MV - {self.profile_dict[index]}"]
                P_load  = net.load[net.load["bus"] == index]["p_mw"].values.tolist()[0]
                profile = df_profiles[self.profile_dict[index]]
                P_bus   = P_load * profile
                Q_load  = net.load[net.load["bus"] == index]["q_mvar"].values.tolist()[0]
                cos_phi = P_load / np.sqrt(P_load ** 2 + Q_load ** 2)
                net.load[net.load["bus"] == index]["p_mw"]   = P_bus
                net.load[net.load["bus"] == index]["q_mvar"] = P_bus * np.sqrt(1 - cos_phi**2) / cos_phi

            elif index in bus_types["PQ_LV"]:
                net.bus.loc[index, ["type", "name"]] = ["b", f"PQ bus {str(index)} - LV - {str(index)}"]
                if not net.load[net.load["bus"] == index]["p_mw"].empty :
                    P_load  = net.load[net.load["bus"] == index]["p_mw"].values.tolist()[0]
                    #TODO Uncomment to follow a profile from file
                    profile = df_profiles[self.profile_dict[index]]
                    P_bus   = P_load * profile
                    #P_bus    = P_load
                    Q_load  = net.load[net.load["bus"] == index]["q_mvar"].values.tolist()[0]
                    #cos_phi = P_load / np.sqrt(P_load ** 2 + Q_load ** 2)
                    cos_phi = 0.95
                    net.load[net.load["bus"] == index]["p_mw"]   = P_bus
                    net.load[net.load["bus"] == index]["q_mvar"] = P_bus * np.sqrt(1 - cos_phi**2) / cos_phi

            else:
                if self.dataset=="MESOGEIA":
                    net.bus.loc[index, ["type", "name"]] = ["b", f"PQ bus {str(index)} - LV - Random {str(index)}"]
                    #net.bus.loc[index, ["type", "name"]] = ["b", f"PQ bus {str(index)} - MV - {self.profile_dict[index]}"]
                    if not net.load[net.load["bus"] == index]["p_mw"].empty :
                        P_load  = net.load[net.load["bus"] == index]["p_mw"].values.tolist()[0]
                        profile = df_profiles[self.profile_dict[index]]
                        P_bus   = P_load * profile
                        Q_load  = net.load[net.load["bus"] == index]["q_mvar"].values.tolist()[0]
                        #cos_phi = P_load / np.sqrt(P_load ** 2 + Q_load ** 2)
                        cos_phi = 0.95
                        net.load[net.load["bus"] == index]["p_mw"]   = P_bus
                        net.load[net.load["bus"] == index]["q_mvar"] = P_bus * np.sqrt(1 - cos_phi**2) / cos_phi


        return net

    def initialize_net(self):

        # Load the MATPOWER case into a Pandapower network
        self.net = pp.converter.from_mpc(self.filepath, f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        self.change_topology(self.net)


        # Retrieve the system base power (Sbase)
        self.S_base = self.net.sn_mva

        #TODO Print Loads and bus types

        self.net = self.assign_buses(self.net, self.dataset)

        # Change P, Q loads ,following random distribution
        self.randomize_loads(self.net)

        # Retrieve the voltage base for each bus (Vbase)
        self.V_base = self.net.bus.vn_kv

        if self.dataset == "MESOGEIA":
            self.net.load["p_mw"]   = 0.50 * self.net.load["p_mw"]
            self.net.load["q_mvar"] = 0.50 * self.net.load["q_mvar"]

    def run_powerflow(self):
        # Run power flow
        pp.runpp(self.net)

    # Compute I branch magnitudes and angles
    # To compute I from per unit -> If_pu = Vpu (from node) / Spu (from node)
    def compute_Ibranch(self):
        # Voltage magnitudes and angles
        vm_pu = self.net.res_bus['vm_pu']  # Voltage magnitudes in per unit
        va_degree = self.net.res_bus['va_degree']  # Voltage angles in degrees

        # Convert to complex voltages in per unit
        V = vm_pu * np.exp(1j * np.radians(va_degree))

        # List to store current magnitude and angle
        branch_currents = []

        for line_idx, line in self.net.line.iterrows():

            from_bus = line['from_bus']  # From bus index
            to_bus = line['to_bus']  # To bus index
            length_km = line['length_km']  # Length of the line (in km)

            # Admittance of the branch (Y = 1 / Z)
            r_ohm_per_km = line['r_ohm_per_km']  # Resistance in ohms per km
            x_ohm_per_km = line['x_ohm_per_km']  # Reactance in ohms per km
            z_ohm = complex(r_ohm_per_km, x_ohm_per_km) * length_km  # Branch impedance

            # Compute base impedance
            Z_base = (self.V_base ** 2) / self.S_base  # Ohms
            Z_base = Z_base.iloc[0]

            Y_branch = 1 / z_ohm  # Branch admittance (in per unit)
            if line['in_service'] == True:
                Y_branch = Y_branch * Z_base
            else:
                Y_branch = 0

            # Voltage difference between the from and to buses (in per unit)
            V_from = V[from_bus]
            V_to = V[to_bus]
            V_diff = V_from - V_to

            # Calculate branch current in per unit (I = Y * V_diff)
            I_branch = Y_branch * V_diff

            # Store magnitude and angle of the branch current
            current_magnitude = abs(I_branch) if line['in_service'] else 0.00
            current_angle = np.angle(I_branch, deg=True) if line['in_service'] else 0.00
            branch_currents.append((line_idx, current_magnitude, current_angle))

        self.net.res_line["Im_pu"] = pd.DataFrame([record[1] for record in branch_currents])
        self.net.res_line["Ia_pu"] = pd.DataFrame([record[2] for record in branch_currents])

    # Just devide with Sbase
    def compute_PQ_pu(self):
        # System base power in MVA
        S_base = self.net.sn_mva
        self.net.res_bus["P_pu"] = self.net.res_bus["p_mw"] / S_base
        self.net.res_bus["Q_pu"] = self.net.res_bus["q_mvar"] / S_base

    # Compute Ybus matrix to get injection currents
    def compute_Inj_currents(self):
        # Voltage magnitudes and angles (in per unit)
        vm_pu = self.net.res_bus['vm_pu']  # Voltage magnitudes in p.u.
        va_degree = self.net.res_bus['va_degree']  # Voltage angles in degrees

        # Convert to complex voltages (in per unit)
        V = vm_pu * np.exp(1j * np.radians(va_degree))
        # Get the system base power from pp_net
        baseMVA = self.net._ppc['baseMVA']

        # Get the bus and branch data from pp_net (they are already in the correct format for PYPOWER)
        bus = self.net._ppc['bus']
        branch = self.net._ppc['branch']

        # Calculate the Ybus matrix using makeYbus
        Ybus, _, _ = makeYbus(baseMVA, bus, branch)

        # Calculate current injections (in per unit)
        I_injections = np.dot(Ybus.toarray(), V)

        Inj_mag = []
        Inj_angle = []

        # Display the current injections in per unit
        for bus_idx, I_inj in enumerate(I_injections):
            Inj_mag.append(abs(I_inj))
            Inj_angle.append(np.angle(I_inj, deg=True))

        self.net.res_bus["Im_inj"] = pd.DataFrame(Inj_mag)
        self.net.res_bus["Ia_inj"] = pd.DataFrame(Inj_angle)


    def get_results(self):
        # Display bus results
        df_V = self.net.res_bus[['vm_pu', 'va_degree', 'P_pu', 'Q_pu', 'Im_inj', 'Ia_inj']]
        df_I = self.net.res_line[["Im_pu", "Ia_pu"]]

        # Insert noise
        df_V, df_I = self.insert_noise(df_V, df_I)

        return pd.concat([df_V, df_I], axis=1)
    def get_powerflow_results(self):

        self.initialize_net()
        self.run_powerflow()
        self.compute_Ibranch()
        self.compute_PQ_pu()
        self.compute_Inj_currents()

        return self.get_results()

# Test topologies
class GenerateDataset:

    def __init__(self, filepath, N_topologies, N_samples, dataset):

        self.filepath = filepath
        self.N_topologies = N_topologies
        self.N_samples = N_samples
        self.dataset = dataset

    def aggregate_random_columns_and_scale(self, df, iteration):

        cols_to_use = [col for col in df.columns if col not in ['utc_timestamp', 'cet_cest_timestamp', 'interpolated']]
        cols_to_use = [col for col in cols_to_use if "LV " not in col]
        # Assign 0 or 1 uniformly to all columns

        # Assign 0 or 1 uniformly at random to each column
        mask = np.random.choice([0, 1], size=len(cols_to_use))

        # Select columns assigned 1
        selected_cols = [col for col, val in zip(cols_to_use, mask) if val == 1]

        # Create new column with sum of selected columns
        df['LV '+str(iteration)] = df[selected_cols].sum(axis=1)

        # Step 2: Generate random weights between 5 and 10
        raw_weights = np.random.uniform(2, 10, size=len(selected_cols))

        # Step 3: Normalize them to sum to 100
        normalized_weights = 100 * raw_weights // raw_weights.sum()

        # Optional: print weights for transparency
        weight_map = dict(zip(selected_cols, normalized_weights))

        # Step 4: Create a new scaled sum column
        df['LV ' + str(iteration)] = sum(df[col] * weight_map[col] for col in selected_cols)

        tmp_df = df[['LV '+str(iteration)]]


        # Normalize to [0, 1]
        min_val = tmp_df['LV '+str(iteration)].min()
        max_val = tmp_df['LV '+str(iteration)].max()

        if max_val == min_val:
            # Avoid division by zero if constant column
            tmp_df['LV '+str(iteration)] = 0.0
            df['LV ' + str(iteration)] = 0.0
        else:
            tmp_df['LV '+str(iteration)] = (tmp_df['LV '+str(iteration)] - min_val) / (max_val - min_val)
            df['LV '+str(iteration)] = (df['LV '+str(iteration)] - min_val) / (max_val - min_val)


        import sys
        #sys.exit(0)

        tmp_df = tmp_df.iloc[24:48]

        # Plot the load profile column
        plt.figure(figsize=(12, 5))
        plt.plot(tmp_df.index, tmp_df['LV ' + str(iteration)], label=f'LV {iteration}')
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.title(f'Load Profile: LV {iteration}')
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(f"load_profiles/LV_{iteration}_load_profile.png", dpi=300, bbox_inches='tight')

        plt.show()

        df.to_csv("datasets/LV_profiles.csv")

        return df

    def generate_LV_profiles(self, num_aggregated_profiles):

        df = pd.read_csv('datasets/household_data.csv')
        df["cet_cest_timestamp"] = df["cet_cest_timestamp"].apply(lambda x: str(x)[:-5].replace("T", " "))
        df = df[df["cet_cest_timestamp"].astype(str).str.contains('2016', na=False)]
        print(df.columns)

        # Set timestamp as index if you want time-based plots
        df.set_index('cet_cest_timestamp', inplace=True)
        df = df.fillna(0)

        load_columns = [col for col in df.columns if col not in ['utc_timestamp', 'cet_cest_timestamp', 'interpolated']]
        for col in load_columns:
            df[col] = df[col].diff()

            # Normalize to [0, 1]
            min_val = df[col].min()
            max_val = df[col].max()

            if max_val == min_val:
                # Avoid division by zero if constant column
                df[col] = 0.0
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)

        for i in range(num_aggregated_profiles):
            self.aggregate_random_columns_and_scale(df, i+1)

        print(df)

    def allign_generated_profiles(self):

        df_alligned = pd.read_csv("datasets/weekly_alligned_scaled_profiles.csv")
        df_lv_profiles = pd.read_csv("datasets/LV_profiles.csv")
        sel_lv_cols = ["cet_cest_timestamp"] + [col for col in df_lv_profiles.columns if "LV " in col]
        df_lv_profiles = df_lv_profiles[sel_lv_cols]
        df_lv_profiles.rename(columns={'cet_cest_timestamp': 'Timestamp'}, inplace=True)
        df_lv_profiles["Timestamp"] = df_lv_profiles["Timestamp"].apply(lambda x: x.replace("2016", "2021"))


        df_alligned = pd.merge(df_alligned, df_lv_profiles, on="Timestamp", how="inner")
        df_alligned.to_csv("datasets/weekly_alligned_scaled_all_profiles.csv")

    def process_profiles(self):

        df_profiles = pd.read_csv(PROFILES_FILEPATH)
        datetime_list = df_profiles["Timestamp"].values.tolist()
        profile_dict = {}

        for i in range(NUM_NODES):
            if i in bus_types["PV_solar"]:
                profile_dict[i] = random.sample(profile_titles["PV"], 1)[0]
            elif i in bus_types["PV_wind"]:
                profile_dict[i] = random.sample(profile_titles["WD"], 1)[0]
            elif i in bus_types["PQ_MV"]:
                profile_dict[i] = random.sample(profile_titles["MV"], 1)[0]
            else:
                profile_dict[i] = random.sample(profile_titles["LV"], 1)[0]

        return datetime_list, profile_dict

    def generate_dataset(self):

        datetime_list, profile_dict = self.process_profiles()
        #print(datetime_list, profile_dict)
        #print(self.dataset)
        concat_frame = pd.DataFrame()
        for i in range(self.N_topologies):
            for j in range(self.N_samples):
                try:
                    datetime_str = datetime_list[j] #TODO Get subset of WD, PV profiles
                    topology = i+1
                    n_sample = j+1
                    print("Topology: ", i+1, "Sample: ", n_sample, "DateTime: ", datetime_str)
                    LPF = LoadPowerFlow(filename=self.filepath,
                                        dataset=self.dataset,
                                        topology=f"""T{topology}""",
                                        datetime_str=datetime_str,
                                        profile_dict=profile_dict)
                    df_results = LPF.get_powerflow_results()
                    df_results["TopNo"] = pd.DataFrame([topology for k in range(df_results.shape[0])])
                    df_results["Simulation"] = pd.DataFrame([n_sample for k in range(df_results.shape[0])])
                    concat_frame = pd.concat([concat_frame, df_results], axis=0)

                except Exception as e:
                    print("Exception"+str(e))

        concat_frame.to_csv(f"datasets/{self.dataset}.csv")



if __name__ == "__main__":
    print(mat_file, NUM_TOPOLOGIES, NUM_SIMULATIONS, dataset)
    GD = GenerateDataset(filepath=mat_file,
                         N_topologies=NUM_TOPOLOGIES,
                         N_samples=NUM_SIMULATIONS,
                         dataset=dataset)
    GD.generate_LV_profiles(num_aggregated_profiles=100)
    GD.allign_generated_profiles()
    GD.generate_dataset()

    #ADP = AllignDataProfiles()
    #solar_file          = ADP.process_PV_data()
    #wind_file           = ADP.process_WD_data()
    #SCADA_file          = ADP.preprocess_MV_DEMAND_data_SCADA()
    #X84_file            = ADP.preprocess_MV_DEMAND_data_X84()
    #residential_file    = ADP.preprocess_LV_DEMAND_data()
    #ADP.allign_datasets()
    #ADP.scale_profiles()
    #ADP.sample_weekly_data()

