import yaml
import pandas as pd

def open_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def open_mimic_meta(config):
    mimic_meta_path = config['path']['mimic_meta_path']
    mimic_split_meta_path = config['path']['mimic_split_meta_path']
    mimic_report_parsed_path = config['path']['mimic_report_parsed_path']
    mimic_meta = pd.read_csv(mimic_meta_path)
    mimic_meta = mimic_meta[(mimic_meta['ViewPosition'] == 'PA') | (mimic_meta['ViewPosition'] == 'AP')]

    mimic_report_parsed = pd.read_csv(mimic_report_parsed_path)    
    # Need Quality Filtering
    mimic_split_meta = pd.read_csv(mimic_split_meta_path)
    return mimic_meta, mimic_split_meta, mimic_report_parsed