import pandas as pd
import os
import shutil

filepaths = {
    "CoLA": "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/base_data/CoLA",
    "MRPC": "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/base_data/MRPC",
    "RTE": "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/base_data/RTE",
    "STS-B": "/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/low_resource/low_resource_glue_data/base_data/STS-B",
}

splits = [1000, 100, 10, 5, 1]

replicates = 10

for key, value in filepaths.items():
    read_header = "infer"
    to_header = True
    if key == "CoLA":
        read_header = None
        to_header = False
    training_data = pd.read_csv(value+"/train.tsv", sep='\t', error_bad_lines=False, header=read_header)
    print("loaded "+key)

    for split in splits:
        split_dir = "split"+str(split)
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)
        for replicate in range(replicates):
            rep_dir = split_dir+"/r"+str(replicate)
            if not os.path.isdir(rep_dir):
                os.mkdir(rep_dir)
            task_dir = rep_dir+"/"+key
            os.mkdir(task_dir)

            sampled_data = training_data.sample(n=split)
            sampled_data.to_csv(task_dir+"/train.tsv", sep="\t", index=False, header=to_header)
            shutil.copy(value+"/test.tsv", task_dir+"/test.tsv")
            shutil.copy(value+"/dev.tsv", task_dir+"/dev.tsv")
