from argparse import ArgumentParser
# To control logging level for various modules used in the application:
import logging
import re

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="models/t5", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-3, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--multi", action='store_true', help="multitask baseline")
    parser.add_argument("--continual", action='store_true', help="continual baseline")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--dataset_list", type=str, default="SGD", help="Path for saving")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", nargs="+", default=[], help="which gpu to use")
    parser.add_argument("--CL", type=str, default="", help="CL strategy used")
    parser.add_argument("--episodic_mem_size", type=int, default=20, help="number of batch put in the episodic memory")
    parser.add_argument("--percentage_LAML", type=float, default=0.2, help="LAML percentage of augmented data used")
    parser.add_argument("--task_type", type=str, default="INTENT", help="Select the kind of task to run")
    parser.add_argument("--bottleneck_size", type=int, default=100, help="lenght of the generation")
    parser.add_argument("--number_of_adpt", type=int, default=25, help="lenght of the generation")
    parser.add_argument("--superposition", action='store_true', help="multitask baseline")
    parser.add_argument("--supsup", action='store_true', help="multitask baseline")
    parser.add_argument("--complex", action='store_true', help="multitask baseline")


    args = parser.parse_args()
    return args


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
