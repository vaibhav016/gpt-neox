import fnmatch
from lm_eval import tasks
import torch
from megatron import mpu



class TaskChecker():
    """
    An adapter to run NeoX models on LM Evaluation Harness (https://github.com/EleutherAI/lm-evaluation-harness) tasks.

    Args:
        model: A NeoX Model
        forward_step_fn: A function that runs a forward pass through the model, returning `tuple(loss, logits)`.
        neox_args: a NeoXArgs object containing the model configuration.
        batch_size (optional): An argument to override the batch size, which defaults to batch size per gpu * dp world size.
    """

    def __init__(self, neox_args):
        self.neox_args = neox_args

        # parallelism args:
        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0



    def process_task_names(self, eval_tasks):
        if eval_tasks is None:
            eval_tasks = [
                "lambada",
                "piqa",
                "hellaswag",
                "winogrande",
                "mathqa",
                "pubmedqa",
            ]
        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        eval_tasks = pattern_match(eval_tasks, tasks.ALL_TASKS)
        print(f"Found tasks: {eval_tasks}")

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.is_local_main:
            task_dict = tasks.get_task_dict(eval_tasks)
        # torch barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        task_dict = tasks.get_task_dict(eval_tasks)


def check_and_download_eval_tasks(neox_args, eval_tasks=None):
    task_checker = TaskChecker(neox_args)
    task_checker.process_task_names(eval_tasks)
    return
    

# Case where this is run without parallelism
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--eval_tasks",
            type=str,
            nargs="+",
            default=None,
            help="Optionally overwrite eval tasks to run for evaluate.py",
        )
    parsed_args = parser.parse_args()

    # Pass on the rank info even if not using parallelism, factoring in that neox args may not be provided in that case.
    dummy_neox_args=type('DummyClass', (object,), {"rank":0, "local_rank":0})()
    check_and_download_eval_tasks(dummy_neox_args, parsed_args.eval_tasks)