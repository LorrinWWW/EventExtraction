import os
import shutil
import json

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.get('map_file', '__NONFILE__')):
        os.remove(params['map_file'])

    if os.path.isdir(params.get('ckpt_path', '__NONFILE__')):
        shutil.rmtree(params['ckpt_path'])

    if os.path.isdir(params.get('summary_path', '__NONFILE__')):
        shutil.rmtree(params['summary_path'])

    if os.path.isdir(params.get('result_path', '__NONFILE__')):
        shutil.rmtree(params['result_path'])

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.get('config_file', '__NONFILE__')):
        os.remove(params['config_file'])

    if os.path.isfile(params.get('vocab_file', '__NONFILE__')):
        os.remove(params['vocab_file'])
        
def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.get('result_path')):
        os.makedirs(params['result_path'])
    if not os.path.isdir(params.get('ckpt_path')):
        os.makedirs(params['ckpt_path'])
    if not os.path.isdir("log"):
        os.makedirs("log")
        
def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)
    