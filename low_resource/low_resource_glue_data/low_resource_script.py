import pandas as pd
import os
import shutil

filepaths = {
    "CoLA": "/home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/low_resource_glue_data/base_data/CoLA",
    "MRPC": "/home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/low_resource_glue_data/base_data/MRPC",
    "RTE": "/home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/low_resource_glue_data/base_data/RTE",
    "STS-B": "/home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/low_resource_glue_data/base_data/STS-B",
}

splits = [1000, 100, 10, 5, 1]

replicates = 10

for key, value in filepaths.items():
    training_data = pd.read_csv(value+"/train.tsv", sep='\t', error_bad_lines=False)
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
            sampled_data = training_data.sample(n=split)
            sampled_data.to_csv(rep_dir+"/train.tsv", sep="\t")
            shutil.copy(value+"/test.tsv", rep_dir+"/test.tsv")
            shutil.copy(value+"/dev.tsv", rep_dir+"/dev.tsv")
