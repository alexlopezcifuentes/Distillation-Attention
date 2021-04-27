import yaml
import os

"""
Utils to read and change the configuration parameters.

Developed by Alejandro Lopez-Cifuentes.
"""


def updateConfig(CONFIG, new_key, new_value):
    """
    Function to replace the value for a given key in CONFIG dictionary.
    Args:
        CONFIG: Full configuration dictionary
        new_key: New key to store the value
        new_value: New value

    Returns: Updated config

    Note: It only supports 3 indent levels in configuration files
    """

    for first_level, second_level in CONFIG.items():
        if type(second_level) is dict:
            for key, value in second_level.items():
                if type(value) is dict:
                    for key2, value2 in value.items():
                        if key2 == new_key:
                            CONFIG[first_level][key][new_key] = str(new_value)
                else:
                    if key == new_key:
                        CONFIG[first_level][new_key] = str(new_value)
        else:
            if first_level == new_key:
                CONFIG[new_key] = str(new_value)

    return CONFIG


def getConfiguration(args):
    """
    Function to join different configuration into one single dictionary.
    Args:
        args:

    Returns: Configuration structure

    """

    default_CONFIG = yaml.safe_load(open(os.path.join('Config', 'config_default.yaml'), 'r'))

    # Check arguments for default configuration
    if args.Dataset is None:
        args.Dataset = default_CONFIG['DEFAULT']['DATASET']
    if args.Architecture is None:
        args.Architecture = default_CONFIG['DEFAULT']['ARCHITECTURE']
    if args.Training is None:
        args.Training = default_CONFIG['DEFAULT']['TRAINING']

    # ----------------------------- #
    #      Dataset Configuration    #
    # ----------------------------- #
    dataset_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Dataset', 'config_' + args.Dataset + '.yaml'), 'r'))

    # ----------------------------- #
    #   Architecture Configuration  #
    # ----------------------------- #
    architecture_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Architecture', 'config_' + args.Architecture + '.yaml'), 'r'))

    # ----------------------------- #
    #    Training Configuration     #
    # ----------------------------- #
    training_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Training', 'config_' + args.Training + '.yaml'), 'r'))

    # ----------------------------- #
    #     Configuration Update      #
    # ----------------------------- #
    # In case there is a configuration update, update configuration
    if args.Options is not None:
        NewOptions = args.Options

        for option in NewOptions:
            new_key, new_value = option.split('=')

            dataset_CONFIG= updateConfig(dataset_CONFIG, new_key, new_value)
            architecture_CONFIG = updateConfig(architecture_CONFIG, new_key, new_value)
            training_CONFIG = updateConfig(training_CONFIG, new_key, new_value)

    # ----------------------------- #
    #      Full Configuration       #
    # ----------------------------- #
    CONFIG = dict()
    CONFIG.update(dataset_CONFIG)
    CONFIG.update(architecture_CONFIG)
    CONFIG.update(training_CONFIG)

    return CONFIG, dataset_CONFIG, architecture_CONFIG, training_CONFIG


def getValidationConfiguration(Model, ResultsPath='Results'):
    """
    Function to join different configuration into one single dictionary.
    Args:
        args:

    Returns: Configuration structure

    """

    Path = os.path.join(ResultsPath, Model)

    # Search for configuration files in model folder
    folder_files = [f for f in os.listdir(Path) if os.path.isfile(os.path.join(Path, f))]

    # Extract only config files
    config_files = [s for s in folder_files if "config_" in s]

    # ----------------------------- #
    #        Configuration 1        #
    # ----------------------------- #
    CONFIG1 = yaml.safe_load(open(os.path.join(Path, config_files[0]), 'r'))

    # ----------------------------- #
    #        Configuration 2        #
    # ----------------------------- #
    CONFIG2 = yaml.safe_load(open(os.path.join(Path, config_files[1]), 'r'))

    # ----------------------------- #
    #        Configuration 3        #
    # ----------------------------- #
    CONFIG3 = yaml.safe_load(open(os.path.join(Path, config_files[2]), 'r'))

    # ----------------------------- #
    #      Full Configuration       #
    # ----------------------------- #
    CONFIG = dict()
    CONFIG.update(CONFIG1)
    CONFIG.update(CONFIG2)
    CONFIG.update(CONFIG3)

    return CONFIG
