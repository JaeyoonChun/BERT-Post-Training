import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seeds
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    init_logger()
    set_seeds()
    tokenizer = load_tokenizer(args)
    if args.do_train:
        trainer = Trainer(args, tokenizer)
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./checkpoints/[path]/best_model.pt", type=str, help="Path to save, load model")
    parser.add_argument("--source_data_dir", default="../data/movie", type=str, help="The source data dir")
    parser.add_argument("--target_data_dir", default="../data/sports", type=str, help="The target data dir")
    parser.add_argument("--test_data_dir", default="../data/finetuning/sports", type=str, help="The test data dir(sports or tv")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="The test data dir(sports or tv")
    parser.add_argument("--num_labels", default=2, type=int, help="numbers of labels")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropot rate")
    parser.add_argument("--hidden_size", default=768, type=int, help="Model hidden size")

    parser.add_argument("--model_name_or_path", default="./checkpoints/[path]", type=str)

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    # args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
