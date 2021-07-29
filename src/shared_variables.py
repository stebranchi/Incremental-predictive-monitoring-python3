"""
This file was created in order to bring
common variables and functions into one file to make
code more clear

"""
import glob
import os

ascii_offset = 161
beam_size = 3
outputs_folder = '../output_files/'
data_folder = '../data/'
#data_folder = '../'
declare_models_folder = '../declare_models/'
epochs = 300
folds = 3
validation_split = 0.2


def get_unicode_from_int(ch):
    return chr(int(ch) + ascii_offset)


def get_int_from_unicode(unch):
    return int(ord(unch)) - ascii_offset


def extract_last_model_checkpoint(log_name, models_folder, fold, model_type):
    model_filepath = outputs_folder + models_folder + '/' + str(fold) + '/models/' + model_type + '/' + log_name + '/'
    list_of_files = glob.glob(model_filepath + '*.h5')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def extract_declare_model_filename(log_name):
    return declare_models_folder + log_name + '.xml'


log_settings = {
    '10x5_1S': {
        'formula': " []( ( \"6\" -> <>( \"3\" ) ) )  /\ <>\"6\" ",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x5_1W': {
        'formula': "<>(\"6\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x5_3S': {
        'formula': " []( ( \"6\" -> <>( \"1\" ) ) )  /\ <>\"6\" /\  []( ( \"8\" -> <>( \"1\" ) ) )  /\ <>\"8\" /\ ",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x5_3W': {
        'formula': "<>(\"8\") /\ <>(\"7\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '5x5_1W': {
        'formula': "<>(\"3\")",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 6
    },
    '5x5_1S': {
        'formula': " []( ( \"3\" -> <>( \"4\" ) ) )  /\ <>\"3\" ",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 6
    },
    '5x5_3W': {
        'formula': "<>(\"4\") /\ <>(\"3\")",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 4
    },
    '5x5_3S': {
        'formula': " []( ( \"4\" -> <>( \"3\" ) ) )  /\ <>\"4\" /\  []( ( \"3\" -> <>( \"0\" ) ) )  /\ <>\"3\" /\ ",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 4
    },
    '10x20_1W': {
        'formula': "<>(\"9\")",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 6
    },
    '10x20_1S': {
        'formula': " []( ( \"8\" -> <>( \"9\" ) ) )  /\ <>\"8\" ",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x20_3W': {
        'formula': "<>(\"9\") /\ <>(\"6\") /\ <>(\"8\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x20_3S': {
        'formula': "[]( ( \"9\" -> <>( \"7\" ) ) )  /\ <>\"9\" /\  []( ( \"8\" -> <>( \"6\" ) ) )  /\ <>\"8\" /\  []( ( \"7\" -> <>( \"5\" ) ) )  /\ <>\"7\" ",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x2_1W': {
        'formula': "<>(\"2\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x2_1S': {
        'formula': " []( ( \"6\" -> <>( \"2\" ) ) )  /\ <>\"6\"",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '10x2_3W': {
        'formula': "<>(\"8\") /\ <>(\"7\")",
        'prefix_size_pred_from': 5,
        'prefix_size_pred_to': 7
    },
    '10x2_3S': {
        'formula': " []( ( \"9\" -> <>( \"3\" ) ) )  /\ <>\"9\" /\  []( ( \"7\" -> <>( \"3\" ) ) )  /\ <>\"7\"",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '50x5_1W': {
        'formula': "<>(\"7\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '50x5_1S': {
        'formula': " []( ( \"7\" -> <>( \"8\" ) ) )  /\ <>\"7\"",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '50x5_3W': {
        'formula': "<>(\"9\") /\ <>(\"7\")",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
    '50x5_3S': {
        'formula': " []( ( \"9\" -> <>( \"7\" ) ) )  /\ <>\"9\" /\  []( ( \"8\" -> <>( \"7\" ) ) )  /\ <>\"8\"",
        'prefix_size_pred_from': 6,
        'prefix_size_pred_to': 7
    },
    'BPI2017_S': {
        'formula': " []( ( \"17\" -> <>( \"18\" ) ) )  /\ <>\"17\"",
        'prefix_size_pred_from': 6,
        'prefix_size_pred_to': 9
    },
    'BPI2017_W': {
        'formula': "<>(\"17\")",
        'prefix_size_pred_from': 6,
        'prefix_size_pred_to': 9
    },
    'BPI2012_S': {
        'formula': " []( ( \"3\" -> <>( \"5\" ) ) )  /\ <>\"3\"",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 5
    },
    'BPI2012_W': {
        'formula': "<>(\"6\")",
        'prefix_size_pred_from': 4,
        'prefix_size_pred_to': 8
    },
    'BPI2012_W_2': {
        'formula': "<>(\"3\")",
        'prefix_size_pred_from': 2,
        'prefix_size_pred_to': 5
    },
}
