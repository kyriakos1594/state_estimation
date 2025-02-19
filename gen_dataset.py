import pandapower as pp
import numpy as np
import pandas as pd
from IEEE_datasets.IEEE33 import config_dict
from pandapower.pypower.makeYbus import makeYbus

# Load Power Flow
class LoadPowerFlow:

    def __init__(self, filename, dataset, topology):

        self.filename = filename
        self.filepath = "datasets/"+self.filename
        self.net      = None
        self.Sbase    = None
        self.Vbase    = None
        self.dataset  = dataset
        self.topology = topology

    def change_topology(self, net):
        if self.dataset == "IEEE33":
            open_branches = config_dict[self.dataset][self.topology]["open_branches"]
            self.net.line["in_service"] = pd.Series(False if (i in open_branches) else True for i in range(self.net.line.shape[0]))
        elif self.dataset == "MESOGEIA":
            open_branches = config_dict[self.dataset][self.topology]["open_branches"]
            self.net.line["in_service"] = pd.Series(False if (i in open_branches) else True for i in range(self.net.line.shape[0]))

    def randomize_loads(self, net):
        self.net.load["LOAD_CHANGE_FLAG"] = np.random.choice([0, 1], size=len(self.net.load))
        r = 0.3
        # var_vector (same size as the number of loads) defines the scaling factors
        # For simplicity, we'll use a vector of ones. Adjust as needed.
        var_vector = np.ones(len(net.load))
        # Apply randomization to active power (p_mw)
        # We now apply the changes to each load based on the load change flag
        net.load["p_mw"] = net.load.apply(
            lambda row: row["p_mw"] * (1 + row["LOAD_CHANGE_FLAG"] * (-r + (2 * r) * np.random.rand())),
            axis=1
        )

        # Υπολογισμός power factor από τις υπάρχουσες τιμές P και Q
        pf = net.load["p_mw"] / np.sqrt(net.load["p_mw"] ** 2 + net.load["q_mvar"] ** 2)
        # Apply randomization to reactive power (q_mvar)
        net.load["q_mvar"] = np.tan(np.arccos(pf)) * net.load["p_mw"]


        #print(self.net.load)
        #print(self.net.bus)

    def insert_noise(self, df_V, df_I):

        # Define constants
        me_phasor_mag = 0.001  # pu
        me_phasor_ang = 0.018  # degrees

        # Insert noise for buses
        # Add measurement noise
        #print("Before: ", df_V)
        df_V['vm_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V['va_degree'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_V)))

        df_V['P_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V.loc[0, 'P_pu'] = 0

        df_V['Q_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V.loc[0, 'Q_pu'] = 0

        df_V['Im_inj'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_V)))
        df_V['Ia_inj'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_V)))
        #print("After: ", df_V)

        # Insert noise for branches
        # Add measurement noise
        # Add noisy magnitude (Im_pu) and angle (Ia_pu in degrees)
        #print("Before: ", df_I)
        df_I['Im_pu'] *= (1 + (me_phasor_mag / 3) * np.random.randn(len(df_I)))
        df_I['Ia_pu'] *= (1 + (me_phasor_ang / 3) * np.random.randn(len(df_I)))
        #print("After: ", df_I)

        return df_V, df_I

    def initialize_net(self):
        # Load the MATPOWER case into a Pandapower network
        self.net = pp.converter.from_mpc(self.filepath, f_hz=50, casename_mpc_file='mpc', validate_conversion=False)

        self.change_topology(self.net)

        #print(self.net.line[["from_bus", "to_bus"]])

        # Retrieve the system base power (Sbase)
        self.S_base = self.net.sn_mva
        #print(f"System Base Power (Sbase): {self.S_base} MVA")

        # Change P, Q loads ,following random distribution
        self.randomize_loads(self.net)

        # Retrieve the voltage base for each bus (Vbase)
        self.V_base = self.net.bus.vn_kv
        #print(f"Voltage Base (Vbase) for Each Bus [kV]: {self.V_base}")

        if self.dataset == "MESOGEIA":
            self.net.load["p_mw"]   = 0.50 * self.net.load["p_mw"]
            self.net.load["q_mvar"] = 0.50 * self.net.load["q_mvar"]

        #print(self.net.load['bus'].values.tolist())

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
            #print("In service: ", line['in_service'])
            if line['in_service'] == True:
                Y_branch = Y_branch * Z_base
            else:
                Y_branch = 0
            #import sys
            #sys.exit(0)

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
        #print("Current Injections (per unit) at each bus:")
        for bus_idx, I_inj in enumerate(I_injections):
            #print(f"Bus {bus_idx}: {I_inj:.6f} p.u. (Magnitude: {abs(I_inj):.6f}, Angle: {np.angle(I_inj, deg=True):.2f} degrees)")
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


        #self.net.res_bus.to_csv("bus.csv")
        #self.net.res_line.to_csv("lines.csv")

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


    def generate_dataset(self):

        concat_frame = pd.DataFrame()

        for i in range(self.N_topologies):
            for j in range(self.N_samples):
                topology = i+1
                print("Topology: ", i+1, "Sample: ", j+1)
                LPF = LoadPowerFlow(filename=self.filepath,
                                    dataset=self.dataset,
                                    topology=f"""T{topology}""")
                df_results = LPF.get_powerflow_results()
                df_results["TopNo"] = pd.DataFrame([topology for k in range(df_results.shape[0])])
                df_results["Simulation"] = pd.DataFrame([j+1 for k in range(df_results.shape[0])])
                concat_frame = pd.concat([concat_frame, df_results], axis=0)

        concat_frame.to_csv(f"datasets/{self.dataset}.csv")

if __name__ == "__main__":
    #PFS = PowerFlowSampler()
    #PFS.generate_results()
    #filename = "Mesogia_demo_site.mat"
    #lpf = LoadPowerFlow(filename, "MESOGEIA", "T2")
    #lpf.initialize_net()
    #lpf.run_powerflow()
    #lpf.compute_Ibranch()
    #lpf.compute_PQ_pu()
    #lpf.compute_Inj_currents()
    #lpf.get_results()
    GD = GenerateDataset(filepath="IEEE33aveg.mat",N_topologies=15,N_samples=1000)
    GD.generate_dataset()

