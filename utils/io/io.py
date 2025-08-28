import sys
import os
from pathlib import Path
import json

def read_file(filepath):
    with open(filepath, "r") as file:
        return file.readlines()

def read_json(filepath):
    with open(filepath, "r") as file:
        jobj = json.load(file)
    return jobj

def read_jsons(filepath):

    with open(filepath, "r") as file:
        json_contents = file.read()
        json_list = json.loads(json_contents)
    return json_list

def read_jsonl(filepath):
    with open(filepath, "r") as file:
        json_list = list(file)
        jobjl = [json.loads(jobj) for jobj in json_list]
    return jobjl

def write_file(filepath, lines):
    with open(filepath, "w") as file:
        file.writelines(lines)

def mkdir(dirpath):
    p = Path(dirpath)
    p.mkdir(exist_ok=True, parents=True)

def os_copy(src, tgt):
    p = Path(src)
    if p.is_file():
        os.system(f"cp {src} {tgt}")

def glob(dirpath, pattern):
    p = Path(dirpath)
    return p.glob(pattern)