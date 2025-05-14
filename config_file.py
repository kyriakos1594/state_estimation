import sys

#TODO Basic configurations
dataset = "95UKGD"
#meterType = "PMU_caseA" # PMU_caseA, PMU_caseB, conventional
#meterType = "PMU_caseB"
#meterType = "conventional"
#GLOBAL_BRANCH_LIST = [6]
#NUM_TOPOLOGIES = 1 #15
NUM_SIMULATIONS = 7787
EXISTING_METER_BRANCHES = []
EXISTING_METER_NODES    = []
RAW_FILENAME    = f"datasets/{dataset}.csv"
MAPE_v_threshold = 0.30
MAE_a_threshold  = 0.15
BATCH_SIZE = 32
PROFILES_FILEPATH = f"datasets/weekly_alligned_scaled_profiles.csv"

#TODO Random Forest Variables
TREE_DEPTH = 10
RF_ESTIMATORS = 25

#IEEE33_PMU_caseA_TI_features      = [6, 33, 27, 10, 32, 5, 4, 20, 8, 24, 7, 25, 19, 26, 9, 34, 17, 3, 2, 11, 12, 16, 18, 31, 23, 13, 15, 14, 22, 21, 29, 30, 28, 1, 0]
#IEEE33_PMU_caseB_TI_features      = [17, 27, 21, 7, 8, 28, 20, 6, 11, 26, 9, 15, 24, 25, 19, 10, 5, 13, 29, 14, 12, 4, 32, 16, 30, 23, 31, 3, 22, 2, 18, 1, 0]
#IEEE33_conventional_TI_features   = [27, 11, 7, 28, 13, 21, 24, 12, 29, 6, 9, 8, 26, 30, 17, 20, 16, 32, 14, 31, 25, 15, 10, 5, 19, 23, 4, 22, 3, 2, 18, 1, 0]

#MESOGEIA_PMU_caseA_TI_features    = []
#MESOGEIA_PMU_caseB_TI_features    = []
#MESOGEIA_conventional_TI_features = []

UKGD95_PMU_caseA_TI_features       = [75, 77, 95, 56, 91, 58, 53, 81, 87, 59, 80, 72, 55, 54, 79, 90, 74, 86, 65, 67, 78, 94, 69, 39, 71, 70, 23, 36, 63, 68, 76, 88, 62, 96, 66, 61, 64, 41, 29, 9, 34, 73, 47, 50, 46, 43, 40, 89, 45, 42, 52, 37, 60, 44, 57, 32, 38, 48, 84, 92, 93, 82, 49, 26, 85, 30, 27, 83, 33, 28, 24, 20, 51, 35, 22, 21, 25, 15, 18, 11, 13, 31, 12, 19, 10, 17, 16, 14, 7, 8, 5, 4, 6, 3, 2, 1, 0]
UKGD95_PMU_caseB_TI_features       = [79, 52, 69, 54, 34, 75, 66, 39, 71, 62, 29, 58, 8, 72, 85, 73, 53, 65, 28, 3, 68, 90, 9, 60, 76, 64, 70, 87, 78, 89, 67, 61, 56, 59, 57, 86, 40, 88, 41, 1, 7, 33, 55, 43, 74, 63, 5, 42, 77, 47, 92, 27, 82, 30, 17, 36, 51, 45, 37, 50, 44, 38, 31, 20, 94, 35, 6, 46, 16, 26, 49, 48, 93, 81, 32, 11, 22, 4, 91, 80, 24, 14, 18, 19, 2, 21, 12, 0, 84, 23, 83, 15, 25, 13, 10]
UKGD95_conventional_TI_features    = [57, 73, 17, 53, 90, 56, 71, 54, 72, 86, 94, 52, 59, 89, 46, 64, 85, 61, 70, 41, 60, 66, 45, 43, 79, 55, 63, 87, 69, 58, 48, 88, 51, 74, 82, 42, 62, 38, 65, 39, 68, 44, 77, 67, 47, 40, 35, 37, 76, 78, 75, 50, 29, 36, 33, 32, 49, 31, 91, 34, 18, 30, 93, 21, 13, 27, 81, 2, 28, 10, 19, 6, 4, 23, 15, 22, 11, 25, 83, 92, 12, 8, 9, 5, 14, 7, 80, 3, 24, 1, 16, 26, 0, 84, 20]

#IEEE33_PMU_caseA_SE_features      = [27, 7, 4, 11, 19, 15, 10, 33, 28, 12, 14, 8, 18, 26, 30, 3, 13, 29, 25, 9, 32, 2, 31, 24, 21, 5, 20, 16, 6, 34, 22, 17, 23, 1, 0]
#IEEE33_PMU_caseB_SE_features      = [27, 7, 11, 8, 14, 10, 15, 29, 12, 28, 30, 20, 26, 13, 9, 5, 2, 21, 32, 25, 31, 4, 17, 24, 22, 16, 6, 23, 19, 3, 0, 1, 18]
#IEEE33_conventional_SE_features   = [24, 27, 11, 15, 28, 10, 13, 7, 32, 16, 9, 29, 20, 17, 31, 21, 14, 8, 30, 12, 6, 26, 25, 5, 23, 4, 19, 3, 2, 1, 22, 18, 0]

#MESOGEIA_PMU_caseA_SE_features    = [18, 16, 11, 22, 12, 23, 9, 64, 7, 39, 28, 52, 66, 5, 31, 51, 27, 42, 114, 36, 29, 122, 131, 93, 83, 98, 47, 85, 77, 116, 128, 41, 2, 72, 99, 124, 46, 123, 132, 44, 119, 126, 129, 110, 109, 59, 97, 111, 80, 130, 69, 102, 48, 115, 90, 103, 82, 121, 112, 104, 107, 89, 113, 92, 1, 75, 108, 88, 78, 106, 91, 100, 105, 84, 117, 86, 79, 61, 125, 50, 73, 63, 49, 67, 118, 76, 71, 65, 62, 87, 0, 101, 57, 95, 68, 96, 56, 37, 40, 25, 35, 120, 74, 14, 70, 15, 32, 10, 58, 8, 45, 81, 24, 21, 94, 17, 26, 13, 54, 38, 55, 4, 33, 30, 43, 34, 20, 6, 53, 19, 3, 60]
#MESOGEIA_PMU_caseB_SE_features    = [68, 57, 80, 67, 46, 59, 58, 76, 102, 97, 50, 52, 53, 121, 36, 79, 41, 43, 98, 56, 49, 42, 61, 64, 93, 63, 30, 48, 91, 60, 39, 37, 116, 65, 105, 73, 81, 74, 72, 101, 8, 100, 122, 70, 23, 35, 108, 45, 112, 83, 28, 84, 103, 87, 124, 95, 118, 128, 32, 130, 126, 99, 125, 85, 69, 51, 75, 113, 96, 110, 94, 88, 34, 104, 129, 117, 107, 86, 71, 106, 38, 115, 111, 109, 24, 21, 89, 114, 78, 92, 29, 62, 66, 119, 90, 54, 0, 47, 33, 15, 10, 40, 77, 12, 22, 82, 55, 13, 18, 9, 27, 19, 25, 44, 11, 31, 5, 26, 3, 6, 2, 16, 20, 14, 4, 17, 7, 1]
#MESOGEIA_conventional_SE_features = [93, 100, 87, 57, 74, 75, 120, 99, 104, 117, 116, 78, 127, 118, 69, 88, 82, 103, 106, 73, 107, 80, 109, 64, 70, 128, 102, 81, 111, 123, 58, 122, 129, 66, 90, 115, 119, 89, 79, 76, 59, 92, 71, 49, 98, 68, 113, 83, 97, 85, 105, 63, 114, 124, 112, 95, 126, 39, 94, 67, 125, 121, 52, 96, 46, 86, 130, 50, 110, 60, 48, 53, 72, 108, 51, 91, 56, 42, 101, 43, 84, 36, 61, 33, 45, 41, 37, 34, 62, 40, 22, 47, 65, 55, 31, 25, 30, 15, 29, 77, 32, 54, 14, 21, 26, 24, 38, 35, 28, 4, 7, 9, 27, 18, 44, 17, 16, 23, 20, 10, 13, 8, 11, 1, 19, 2, 6, 12, 3, 5, 0]

UKGD95_PMU_caseA_SE_features       = [67, 65, 84, 80, 64, 83, 95, 77, 70, 66, 55, 59, 82, 71, 81, 61, 69, 39, 79, 76, 63, 87, 54, 72, 78, 24, 62, 8, 36, 56, 41, 60, 96, 34, 91, 58, 74, 93, 29, 25, 90, 75, 23, 68, 6, 40, 86, 43, 21, 88, 92, 22, 32, 45, 53, 51, 44, 89, 47, 26, 49, 9, 33, 46, 37, 15, 52, 2, 42, 94, 4, 11, 27, 30, 7, 13, 18, 38, 35, 19, 31, 14, 48, 12, 16, 20, 50, 10, 0, 5, 28, 3, 1, 85]
UKGD95_PMU_caseB_SE_features       = [65, 62, 94, 79, 75, 90, 57, 63, 69, 82, 58, 76, 56, 81, 70, 34, 77, 68, 86, 39, 20, 60, 89, 72, 78, 59, 67, 52, 85, 54, 45, 61, 29, 71, 28, 17, 74, 53, 87, 30, 43, 14, 64, 49, 55, 8, 92, 91, 40, 66, 46, 80, 73, 12, 42, 41, 31, 38, 7, 44, 15, 21, 88, 47, 9, 4, 93, 50, 1, 35, 32, 36, 5, 37, 48, 16, 24, 51, 11, 6, 3, 13, 27, 23, 25, 26, 10, 22, 19, 2, 33, 18, 0, 84, 83]
UKGD95_conventional_SE_features    = [94, 82, 57, 90, 17, 73, 89, 52, 79, 88, 86, 39, 56, 41, 46, 87, 45, 69, 48, 68, 19, 42, 18, 38, 85, 72, 44, 40, 21, 70, 53, 50, 43, 54, 49, 51, 36, 66, 55, 20, 47, 35, 37, 78, 67, 59, 31, 33, 71, 77, 76, 91, 64, 60, 74, 61, 83, 2, 65, 34, 7, 30, 10, 63, 29, 93, 32, 22, 15, 23, 27, 81, 58, 6, 0, 13, 4, 25, 16, 8, 28, 62, 9, 11, 5, 3, 92, 84, 24, 1, 80, 26, 12, 14]

if dataset == "MESOGEIA":
    # MESOGEIA
    NUM_NODES = 131
    NUM_BRANCHES = 133
    NUM_TOPOLOGIES = 2
    NUM_SAMPLES = 7500
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
    NODE_PICK_LIST = [4, 7, 9, 14, 15, 17, 18, 20, 21, 22, 25, 26, 27, 31, 33, 37, 40, 45, 53, 59, 60, 62, 68, 70, 75,
                      76, 77, 79, 80, 82, 83, 85, 88, 89, 90, 92, 94, 95, 96, 98, 102, 104, 105, 107, 108, 109, 111,
                      112, 114, 115, 117, 118, 119, 121, 124, 127, 128, 129, 130]
    BRANCH_PICK_LIST = [key for key in branch_data.keys() if (
                (branch_data[key]["receiving_node"] in NODE_PICK_LIST) or (
                    branch_data[key]["sending_node"] in NODE_PICK_LIST))]

    print(BRANCH_PICK_LIST)

    bus_types = {
        "slack": [0],
        "PV_wind": [],
        "PV_solar": [61, 65],
        "PQ_MV": [],
        "PQ_LV": [i for i in range(NUM_NODES) if i not in [0, 61, 65]],
    }

    profile_titles = {
        "PV": ["PV_profile 1", "PV_profile 2", "PV_profile 3", "PV_profile 4", "PV_profile 5", "PV_profile 6", "PV_profile 7"],
        "WD": ["WD 1"],
        "MV": ["R210 Load","X84_1 MV Load_x","X84_1 MV Load_y"],
        "LV": ["R1", "R2", "R3", "R4", "R5", "R6"]
    }

elif dataset == "95UKGD":
    mat_file = "95UKGDaveg.mat"
    NUM_NODES = 95
    NUM_BRANCHES = 97
    NUM_TOPOLOGIES = 8
    NUM_SAMPLES = 7787
    NUM_SIMULATIONS = NUM_SAMPLES
    branch_data = {
        0: {"sending_node": 0, "receiving_node": 1},
        1: {"sending_node": 0, "receiving_node": 84},
        2: {"sending_node": 1, "receiving_node": 3},
        3: {"sending_node": 2, "receiving_node": 3},
        4: {"sending_node": 3, "receiving_node": 5},
        5: {"sending_node": 4, "receiving_node": 5},
        6: {"sending_node": 5, "receiving_node": 7},
        7: {"sending_node": 6, "receiving_node": 7},
        8: {"sending_node": 7, "receiving_node": 9},
        9: {"sending_node": 8, "receiving_node": 9},
        10: {"sending_node": 8, "receiving_node": 27},
        11: {"sending_node": 8, "receiving_node": 28},
        12: {"sending_node": 9, "receiving_node": 10},
        13: {"sending_node": 10, "receiving_node": 12},
        14: {"sending_node": 11, "receiving_node": 12},
        15: {"sending_node": 12, "receiving_node": 14},
        16: {"sending_node": 13, "receiving_node": 14},
        17: {"sending_node": 14, "receiving_node": 16},
        18: {"sending_node": 15, "receiving_node": 16},
        19: {"sending_node": 16, "receiving_node": 24},
        20: {"sending_node": 17, "receiving_node": 18},
        21: {"sending_node": 18, "receiving_node": 20},
        22: {"sending_node": 19, "receiving_node": 20},
        23: {"sending_node": 20, "receiving_node": 21},
        24: {"sending_node": 21, "receiving_node": 22},
        25: {"sending_node": 22, "receiving_node": 23},
        26: {"sending_node": 23, "receiving_node": 24},
        27: {"sending_node": 24, "receiving_node": 26},
        28: {"sending_node": 25, "receiving_node": 26},
        29: {"sending_node": 28, "receiving_node": 29},
        30: {"sending_node": 28, "receiving_node": 30},
        31: {"sending_node": 29, "receiving_node": 31},
        32: {"sending_node": 29, "receiving_node": 33},
        33: {"sending_node": 31, "receiving_node": 32},
        34: {"sending_node": 33, "receiving_node": 34},
        35: {"sending_node": 34, "receiving_node": 35},
        36: {"sending_node": 34, "receiving_node": 38},
        37: {"sending_node": 35, "receiving_node": 36},
        38: {"sending_node": 36, "receiving_node": 37},
        39: {"sending_node": 38, "receiving_node": 39},
        40: {"sending_node": 39, "receiving_node": 40},
        41: {"sending_node": 39, "receiving_node": 52},
        42: {"sending_node": 40, "receiving_node": 41},
        43: {"sending_node": 40, "receiving_node": 42},
        44: {"sending_node": 42, "receiving_node": 43},
        45: {"sending_node": 43, "receiving_node": 44},
        46: {"sending_node": 44, "receiving_node": 45},
        47: {"sending_node": 45, "receiving_node": 46},
        48: {"sending_node": 45, "receiving_node": 47},
        49: {"sending_node": 47, "receiving_node": 48},
        50: {"sending_node": 47, "receiving_node": 49},
        51: {"sending_node": 49, "receiving_node": 50},
        52: {"sending_node": 49, "receiving_node": 51},
        53: {"sending_node": 52, "receiving_node": 53},
        54: {"sending_node": 52, "receiving_node": 54},
        55: {"sending_node": 53, "receiving_node": 58},
        56: {"sending_node": 53, "receiving_node": 74},
        57: {"sending_node": 54, "receiving_node": 55},
        58: {"sending_node": 54, "receiving_node": 56},
        59: {"sending_node": 56, "receiving_node": 57},
        60: {"sending_node": 58, "receiving_node": 59},
        61: {"sending_node": 58, "receiving_node": 61},
        62: {"sending_node": 59, "receiving_node": 60},
        63: {"sending_node": 61, "receiving_node": 62},
        64: {"sending_node": 62, "receiving_node": 63},
        65: {"sending_node": 62, "receiving_node": 64},
        66: {"sending_node": 64, "receiving_node": 65},
        67: {"sending_node": 65, "receiving_node": 66},
        68: {"sending_node": 66, "receiving_node": 67},
        69: {"sending_node": 66, "receiving_node": 68},
        70: {"sending_node": 68, "receiving_node": 69},
        71: {"sending_node": 69, "receiving_node": 70},
        72: {"sending_node": 70, "receiving_node": 71},
        73: {"sending_node": 71, "receiving_node": 72},
        74: {"sending_node": 71, "receiving_node": 73},
        75: {"sending_node": 74, "receiving_node": 75},
        76: {"sending_node": 75, "receiving_node": 76},
        77: {"sending_node": 75, "receiving_node": 79},
        78: {"sending_node": 76, "receiving_node": 77},
        79: {"sending_node": 77, "receiving_node": 78},
        80: {"sending_node": 79, "receiving_node": 82},
        81: {"sending_node": 79, "receiving_node": 85},
        82: {"sending_node": 80, "receiving_node": 81},
        83: {"sending_node": 80, "receiving_node": 93},
        84: {"sending_node": 81, "receiving_node": 94},
        85: {"sending_node": 83, "receiving_node": 84},
        86: {"sending_node": 85, "receiving_node": 86},
        87: {"sending_node": 85, "receiving_node": 89},
        88: {"sending_node": 86, "receiving_node": 87},
        89: {"sending_node": 87, "receiving_node": 88},
        90: {"sending_node": 89, "receiving_node": 90},
        91: {"sending_node": 90, "receiving_node": 91},
        92: {"sending_node": 91, "receiving_node": 92},
        93: {"sending_node": 92, "receiving_node": 93},
        94: {"sending_node": 17, "receiving_node": 34},
        95: {"sending_node": 57, "receiving_node": 82},
        96: {"sending_node": 91, "receiving_node": 73}
    }
    NODE_PICK_LIST = []
    BRANCH_PICK_LIST = [key for key in branch_data.keys() if (
            (branch_data[key]["receiving_node"] in NODE_PICK_LIST) or (
            branch_data[key]["sending_node"] in NODE_PICK_LIST))]

    print(BRANCH_PICK_LIST)

    bus_types = {
        "slack": [0],
        "PV_wind": [17],
        "PV_solar": [94],
        "PQ_MV": [83, 91, 18, 86, 88, 73, 93],
        "PQ_LV": [i for i in range(NUM_NODES) if i not in [0, 17, 94, 83, 91, 18, 86, 88, 73, 93]],
    }

    profile_titles = {
        "PV": ["PV_profile 1", "PV_profile 2", "PV_profile 3", "PV_profile 4", "PV_profile 5",
               "PV_profile 6", "PV_profile 7"],
        "WD": ["WD 1"],
        "MV": ["R210 Load", "X84_1 MV Load_x", "X84_1 MV Load_y"],
        "LV": ["R1", "R2", "R3", "R4", "R5", "R6"]
    }

elif dataset == "IEEE33":
    mat_file = "IEEE33aveg.mat"
    NUM_NODES = 33  # 131 #33
    NUM_BRANCHES = 35  # 133 #35
    NUM_TOPOLOGIES = 15
    NUM_SAMPLES = 1000
    NUM_SIMULATIONS = NUM_SAMPLES
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
    NODE_PICK_LIST = []
    BRANCH_PICK_LIST = []

    bus_types = {
        "slack": [0],
        "PV_wind": [32],
        "PV_solar": [17, 21],
        "PQ_MV": [3, 13, 23, 24, 28, 29, 30, 31],
        "PQ_LV": [i for i in range(NUM_NODES) if i not in [0, 17, 21, 32, 3, 13, 23, 24, 28, 29, 30, 31]],
    }

    profile_titles = {
        "PV": ["PV_profile 1", "PV_profile 2", "PV_profile 3", "PV_profile 4", "PV_profile 5", "PV_profile 6", "PV_profile 7"],
        "WD": ["WD 1"],
        "MV": ["R210 Load","X84_1 MV Load_x","X84_1 MV Load_y"],
        "LV": ["R1", "R2", "R3", "R4", "R5", "R6"]
    }

else:
    print("Enter known dataset")
    sys.exit(0)

#TODO Data files for PMU case A
PMU_caseA_dataset           = f"datasets/{dataset}PMU_caseA_dataset.csv"
PMU_caseA_input             = f"datasets/{dataset}PMU_caseA_input.npy"
PMU_caseA_output            = f"datasets/{dataset}PMU_caseA_output.npy"
X_train_PMU_caseA           = f"datasets/{dataset}_PMU_caseA_X_train.npy"
y_train_PMU_caseA           = f"datasets/{dataset}_PMU_caseA_y_train.npy"
X_val_PMU_caseA             = f"datasets/{dataset}_PMU_caseA_X_val.npy"
y_val_PMU_caseA             = f"datasets/{dataset}_PMU_caseA_y_val.npy"
X_test_PMU_caseA            = f"datasets/{dataset}_PMU_caseA_X_test.npy"
X_test_PMU_caseA_outliers   = f"datasets/{dataset}_PMU_caseA_X_test_outliers.npy"
X_test_PMU_caseA_imputed    = f"datasets/{dataset}_PMU_caseA_X_test_imputed.npy"
y_test_PMU_caseA            = f"datasets/{dataset}_PMU_caseA_y_test.npy"
y_test_PMU_caseA_imputed    = f"datasets/{dataset}_PMU_caseA_y_test_imputed.npy"

#TODO Data files for PMU case B
PMU_caseB_dataset = f"datasets/{dataset}PMU_caseB_dataset.csv"
PMU_caseB_input   = f"datasets/{dataset}PMU_caseB_input.npy"
PMU_caseB_output  = f"datasets/{dataset}PMU_caseB_output.npy"
X_train_PMU_caseB = f"datasets/{dataset}_PMU_caseB_X_train.npy"
y_train_PMU_caseB = f"datasets/{dataset}_PMU_caseB_y_train.npy"
X_val_PMU_caseB   = f"datasets/{dataset}_PMU_caseB_X_val.npy"
y_val_PMU_caseB   = f"datasets/{dataset}_PMU_caseB_y_val.npy"
X_test_PMU_caseB  = f"datasets/{dataset}_PMU_caseB_X_test.npy"
y_test_PMU_caseB  = f"datasets/{dataset}_PMU_caseB_y_test.npy"

#TODO Data files for conventional meters
conventional_dataset = f"datasets/{dataset}_conventional_dataset.csv"
conventional_input   = f"datasets/{dataset}_conventional_input.npy"
conventional_output  = f"datasets/{dataset}_conventional_output.npy"
X_train_conventional = f"datasets/{dataset}_conventional_X_train.npy"
y_train_conventional = f"datasets/{dataset}_conventional_y_train.npy"
X_val_conventional   = f"datasets/{dataset}_conventional_X_val.npy"
y_val_conventional   = f"datasets/{dataset}_conventional_y_val.npy"
X_test_conventional  = f"datasets/{dataset}_conventional_X_test.npy"
y_test_conventional  = f"datasets/{dataset}_conventional_y_test.npy"
