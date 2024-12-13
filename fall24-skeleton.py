import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np
import copy
import datetime
import itertools

import helper_funcs
from helper_funcs import Node, Root


if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.

    f = open(DGH_file,"r")
    mystring= f.readlines()
    folder = "DGHs/"
    f_extension = ".txt"
    domain = f.name[len(folder):-len(f_extension)]

    root = Root(domain)
    for_tab_c = 0


    for e in mystring:
        tab_c = 0

        if e[0] != "\t":
            e= e.replace("\n","")
            node = Node(e,0)
            root.child = node

            for_tab_c +=1

        else:
            for i in e:
                if i == "\t":
                    tab_c +=1

            current = root.elevator(tab_c)

            nodeName= e.replace("\n","")
            nodeName = nodeName.replace("\t","")
            node = Node(nodeName, tab_c, current)
            current.child.append(node)
    return root

    
   # with open(DGH_file) as f:
    #    pass


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str, DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    # Remove any processing columns like 'check' if they exist
    for record in anonymized_dataset:
        if 'check' in record:
            del record['check']
    
    # Strip whitespace from all string data
    for dataset in [raw_dataset, anonymized_dataset]:
        for record in dataset:
            for key in record:
                if isinstance(record[key], str):
                    record[key] = record[key].strip()
    
    
    # Assertion
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = helper_funcs.add_costMD_col(anonymized_dataset)
    
    name_list_domain = list(DGHs.values())

    anonymized_dataset = helper_funcs.find_value_of_md(name_list_domain,raw_dataset, anonymized_dataset)
    
    c =helper_funcs.find_table_md(anonymized_dataset)


    return c


     


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str, DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    # Remove the 'check' column if it exists in the anonymized dataset
    for record in anonymized_dataset:
        if 'check' in record:
            del record['check']
    
    
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    
    DGHs = read_DGHs(DGH_folder)
    anonymized_dataset = helper_funcs.add_costLM_col(anonymized_dataset)
    name_list_domain = list(DGHs.values())

    l = len(name_list_domain)

    anonymized_dataset = helper_funcs.find_value_lm(name_list_domain, anonymized_dataset)
    helper_funcs.find_records_lm(anonymized_dataset, l)
    c = helper_funcs.find_table_lm(anonymized_dataset)

    return c


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      s: int, output_file: str):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    EC_list = helper_funcs.find_ECs(raw_dataset, k)
    clusters= helper_funcs.generalize_data(DGHs, EC_list)
    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    general_cost_dict = helper_funcs.create_generalization_cost_dict(DGHs)
    anonymized_dataset = helper_funcs.add_check_col(raw_dataset)
    anonymized_dataset = helper_funcs.add_cost_col(anonymized_dataset)
    anonymized_dataset = helper_funcs.add_index_col(anonymized_dataset)

    for r in anonymized_dataset:

        if r["check"]:
            continue
        r["check"]=True

        customized_ds = list()
        customized_ds.append(r)

        for anon in anonymized_dataset:

            if not anon["check"]:
                customized_ds.append(helper_funcs.calculate_generalization_cost(DGHs, general_cost_dict,r,anon)) 
        
        helper_funcs.find_min_generalization_cost(DGHs, customized_ds, k)

    helper_funcs.del_meanless_cols(anonymized_dataset)


    write_dataset(anonymized_dataset, output_file)


def bottom_up_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, l: int, output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        l (int): distinct l-diversity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = deepcopy(raw_dataset)
    
    for record in anonymized_dataset:
        record['check'] = False
    
  
    current_level = 0
    max_level = helper_funcs.find_max_h_DGHs(DGHs)
    
  
    while not helper_funcs.k_anonymity_check(DGHs, anonymized_dataset, k) and current_level < max_level:
        
        for record in anonymized_dataset:
            helper_funcs.generalize_record(record, DGHs, current_level)
        
        current_level += 1
    
    if not helper_funcs.k_anonymity_check(DGHs, anonymized_dataset, k):
        print("No k-anonymized dataset found.")
        return
    
    write_dataset(anonymized_dataset, output_file)
    

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, bottom_up]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottom_up']:
    print("Invalid algorithm.")
    sys.exit(2)
###### to measure time
start = datetime.datetime.now()
print(start)
#########
dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function != clustering_anonymizer:
    if len(sys.argv) != 7:
        print(
            f"Usage: python3 {sys.argv[0]} <algorithm name> DGH-folder raw-dataset.csv k anonymized.csv seed/l(random/bottom_up)")
        print(f"\tWhere algorithm is one of [clustering, random, bottom_up]")
        sys.exit(1)

    last_param = int(sys.argv[6])
    function(raw_file, dgh_path, k, last_param, anonymized_file)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

####to measure time
end = datetime.datetime.now()
print(end)

print(f'Time taken: {end-start}')
##############
# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300

