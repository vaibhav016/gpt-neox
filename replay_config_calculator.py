import os
import subprocess

div = int(265/7) * 1000

dataset_names = ["pile", "CommonCrawl", "C4", "Github", "ArXiv", "Book", "Wikipedia", "StackExchange"]
dataset_metadata = {}   
dataset_metadata["pile"] = {"size": 299.28e9}
dataset_metadata["CommonCrawl"] = {"size": 155887114486}
dataset_metadata["C4"] = {"size": 79872895846}
dataset_metadata["Github"] = {"size": 15634722173}
dataset_metadata["ArXiv"] = {"size": 13252597099}
dataset_metadata["Book"] = {"size": 12579061292}
dataset_metadata["Wikipedia"] = {"size": 11963032160}
dataset_metadata["StackExchange"] = {"size": 10088908154}


class IndividualTaskConfigHelper():
    def __init__(self, current_task_name, task_number, iterations, replay_percent, seq_len) -> None:
        self.current_task_name = current_task_name
        self.task_number = task_number
        self.replay_percent = replay_percent
        self.num_seen_seq_during = {}
        self.num_seen_tokens_during = {}
        self.cum_num_seen_seq_at_start = {}
        self.cum_num_seen_seq_at_end = None
        self.offsets = {}
        # weights pre multiplication by replay percent
        self.replay_weights = {}
        # 0.5% buffer. 
        self.iterations = iterations
        pass

    def init_all_attributes(self, batch_size, seq_len, prev_config=None):
        # set replay to 0 if we have no replay
        if len(self.cum_num_seen_seq_at_start.values()) == 0:
            self.replay_percent = 0
        self.init_replay_weights()
        self.init_num_seen_seq_during(batch_size, seq_len)
        self.init_num_seen_tokens_during(seq_len)
        self.init_offsets(prev_config)
        self.init_cum_num_seen_seq_at_end()

    def init_replay_weights(self):
        if len(self.cum_num_seen_seq_at_start.values()) == 0:
            return
        sum_weights = sum(self.cum_num_seen_seq_at_start.values())
        if sum_weights == 0:
            return
        for k, v in self.cum_num_seen_seq_at_start.items():
            self.replay_weights[k] = v / sum_weights
        return

    def init_num_seen_seq_during(self, batch_size, seq_len):
        for k in self.replay_weights.keys():
            self.num_seen_seq_during[k] = int(self.iterations * batch_size * self.replay_weights[k] * self.replay_percent)
        self.num_seen_seq_during[self.current_task_name] = int(self.iterations * batch_size * (1 - self.replay_percent))
        return
    
    def init_num_seen_tokens_during(self, seq_len):
        for k, v in self.num_seen_seq_during.items():
            self.num_seen_tokens_during[k] = v * seq_len
        return
            

    # leave a margin of error of offsets due to random sampling potentially having sampled one dataset
    # a bit more than what the weights that were used would suggest
    def init_offsets(self, prev_config, margin_of_error=1.005):
        if prev_config is not None:
            for k, v in prev_config.offsets.items():
                self.offsets[k] = v + int(margin_of_error * prev_config.num_seen_seq_during[k])
            self.offsets[prev_config.current_task_name] = 0 #int(margin_of_error * self.num_seen_seq_during[self.current_task_name])
        else:
            self.offsets = {}
        return

    def init_cum_num_seen_seq_at_end(self):
        self.cum_num_seen_seq_at_end = {}
        for k, v in self.cum_num_seen_seq_at_start.items():
            self.cum_num_seen_seq_at_end[k] = v + self.num_seen_seq_during[k]
        self.cum_num_seen_seq_at_end[self.current_task_name] = self.num_seen_seq_during[self.current_task_name]
        return
        



def get_dataset_configs(dataset_metadata, batch_size, seq_len = 2048, replay_percent=0.05):
    dataset_configs = []

    for i, dataset_name in enumerate(dataset_names):
        iterations = dataset_metadata[dataset_name].get("iterations", int((dataset_metadata[dataset_name]["size"] // seq_len / 1.000)) // batch_size)
        print(iterations)
        current_config = IndividualTaskConfigHelper(current_task_name=dataset_name, 
                                                    task_number=i, 
                                                    iterations=iterations,
                                                    replay_percent=replay_percent,
                                                    seq_len=seq_len
                                                    )

        if i > 0:
            prev_config = dataset_configs[-1]
            current_config.cum_num_seen_seq_at_start = prev_config.cum_num_seen_seq_at_end
            current_config.init_all_attributes(batch_size, seq_len, prev_config=prev_config)
        else:
            current_config.init_all_attributes(batch_size, seq_len, prev_config=None)

        
        print(vars(current_config))
        dataset_configs.append(current_config)
        
    return dataset_configs

get_dataset_configs(dataset_metadata=dataset_metadata, batch_size=1104, replay_percent=0.05)



# for x in range(7):

#     script = """#!/bin/bash
# #SBATCH -A CSC499
# #SBATCH -J {}_to_{}_9B_SP_scratch
# #SBATCH -N 1
# #SBATCH -t 24:00:00
# ##SBATCH --mail-user ibrahima@mila.quebec
# ##SBATCH --mail-type=END
# #SBATCH -o /gpfs/alpine/csc499/scratch/adami/dtn_logs/out.%j    
# #SBATCH -e /gpfs/alpine/csc499/scratch/adami/dtn_logs/out.%j    

# source /gpfs/alpine/csc499/scratch/$(whoami)/andes_install/setup.sh

# cd /gpfs/alpine/csc499/scratch/$(whoami)/andes_install
# python cp_ckpts.py --start {} --end {} --filter '7-1B,pile+slim-pajama-300B-each'""".format(x*div , div*(x+1), x*div, div*(x+1))
#     with open('cp_{}.sh'.format(x), 'w') as f:
#         f.write(script)

#     # sbatch_command = ["sbatch", "cp_{}.sh".format(x)]
#     # subprocess.Popen(sbatch_command)