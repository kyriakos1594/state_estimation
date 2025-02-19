import sys

#TODO Basic configurations
dataset = "IEEE33"
#meterType = "PMU_caseA" # PMU_caseA, PMU_caseB, conventional
meterType = "PMU_caseB"
#meterType = "conventional"
GLOBAL_BRANCH_LIST = [6]
NUM_TOPOLOGIES = 15
NUM_SIMULATIONS = 1000
EXISTING_METER_BRANCHES = []
EXISTING_METER_NODES    = []
RAW_FILENAME    = f"datasets/{dataset}.csv"


if dataset == "MESOGEIA":
    # MESOGEIA
    NUM_NODES = 131
    NUM_BRANCHES = 133
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
elif dataset == "IEEE33":
    NUM_NODES = 33  # 131 #33
    NUM_BRANCHES = 35  # 133 #35
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
else:
    print("Enter known dataset")
    sys.exit(0)

#TODO Data files for PMU case A
PMU_caseA_dataset = f"datasets/{dataset}PMU_caseA_dataset.csv"
PMU_caseA_input   = f"datasets/{dataset}PMU_caseA_input.npy"
PMU_caseA_output  = f"datasets/{dataset}PMU_caseA_output.npy"
X_train_PMU_caseA = f"datasets/{dataset}_PMU_caseA_X_train.npy"
y_train_PMU_caseA = f"datasets/{dataset}_PMU_caseA_y_train.npy"
X_val_PMU_caseA   = f"datasets/{dataset}_PMU_caseA_X_val.npy"
y_val_PMU_caseA   = f"datasets/{dataset}_PMU_caseA_y_val.npy"
X_test_PMU_caseA  = f"datasets/{dataset}_PMU_caseA_X_test.npy"
y_test_PMU_caseA  = f"datasets/{dataset}_PMU_caseA_y_test.npy"

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