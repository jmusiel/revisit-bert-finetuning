from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import os
import logging
import numpy as np
import random
from argparse import Namespace
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

def main():
    base_dir_path = "/home/jovyan/joe-cls-vol/class_projects/nlp_11711_project/revisit-bert-finetuning"

    model = AutoModelForSequenceClassification.from_pretrained(base_dir_path+"/low_resource/bert_output_model_test/split1/r0/reinit_debiased/MRPC/SEED0/checkpoint-last")
    model = model.eval()
    test_set_path = base_dir_path+"/low_resource/low_resource_glue_data/base_data/MRPC/test.tsv"

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
            'bert-large-uncased',
            do_lower_case=True,
            cache_dir=base_dir_path+"/cache",
        )

    args = Namespace(
        local_rank=-1,
        data_dir=base_dir_path+"/low_resource/low_resource_glue_data/base_data/MRPC",
        resplit_val=0,
        model_name_or_path='bert-large-uncased',
        max_seq_length=128,
        overwrite_cache=False,
        model_type="bert",
        downsample_trainset=-1,
        seed=0
        )
    task = "mrpc"
    eval_dataset = load_and_cache_examples(args, task, tokenizer, evaluate=True)

    for seed in [0, 1, 2, 3, 4]:
        seed_str = "SEED" + str(seed)
        for replicate in [0, 1, 2, 3, 4]:
            r_str = "r" + str(replicate)
            model = AutoModelForSequenceClassification.from_pretrained(base_dir_path+"/low_resource/bert_output_model_test/split1/"+r_str+"/reinit_debiased/MRPC/"+seed_str+"/checkpoint-last")
            model = model.eval()

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=10)

            outputList = []
            tot_correct = 0
            num_ex = 0
            for batch in eval_dataloader:
                print("batch (labels, predicted, num_correct, acc)")
                print("labels:")
                print(batch[3])
                # batch = tuple(t.to("cpu") for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                    # "labels": batch[3],
                }
                outputs = model(**inputs)[0]
                _, preds = torch.max(outputs, dim=1)
                print("predictions:")
                print(preds)
                correct = torch.sum(preds == batch[3])
                acc  = correct / preds.shape[0]
                tot_correct += correct
                num_ex += preds.shape[0]
                print("num_correct:")
                print(correct)
                print("accuracy:")
                print(acc)
            print("done")
            print(tot_correct)
            print(num_ex)
            print(tot_correct / num_ex)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if (evaluate and args.resplit_val <= 0) else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.downsample_trainset > 0 and not evaluate:
        assert (args.downsample_trainset + args.resplit_val) <= len(features)

    if args.downsample_trainset > 0 or args.resplit_val > 0:
        set_seed(0)  # use the same seed for downsample
        if output_mode == "classification":
            label_to_idx = defaultdict(list)
            for i, f in enumerate(features):
                label_to_idx[f.label].append(i)

            samples_per_class = args.resplit_val if evaluate else args.downsample_trainset
            samples_per_class = samples_per_class // len(label_to_idx)

            for k in label_to_idx:
                label_to_idx[k] = np.array(label_to_idx[k])
                np.random.shuffle(label_to_idx[k])
                if evaluate:
                    if args.resplit_val > 0:
                        label_to_idx[k] = label_to_idx[k][-samples_per_class:]
                    else:
                        pass
                else:
                    if args.resplit_val > 0 and args.downsample_trainset <= 0:
                        samples_per_class = len(label_to_idx[k]) - args.resplit_val // len(label_to_idx)
                    label_to_idx[k] = label_to_idx[k][:samples_per_class]

            sampled_idx = np.concatenate(list(label_to_idx.values()))
        else:
            if args.downsample_trainset > 0:
                sampled_idx = torch.randperm(len(features))[: args.downsample_trainset]
            else:
                raise NotImplementedError
        set_seed(args.seed)
        features = [features[i] for i in sampled_idx]

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

if __name__ == "__main__":
    main()