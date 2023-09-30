import os
import copy
import pprint
import timeit
import argparse
import multiprocessing
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300 
from functools import reduce
import re
step_to_token = lambda x : x * 2260992/1000000000

fontdict = {'family': 'serif',
#              'color':  'black',
             'weight': 'normal',
             'size': 12,}

plt.rc('font', **fontdict)


legend_font = {'family': 'serif',
                     'color':  'black',
                     'weight': 'normal',
                     'size': 12,}

cm = {'Point-Baseline':'red',
      'Point-Transformer':'darkred',
      'Point-Transformer2M 3200e':'#8b4513',
      'Point-Transformer2M 1600e':'#d2691e',
      'Point-Transformer2M 800e':'#cd853f',
      'Point-Transformer2M 400e':'#f4a460',
      'Point-Transformer2M 200e':'#ffe4c4',
      'DGCNN':'darkblue',
      'PointNet':'darkorange',
      'Deit-Tiny PT': 'darkgreen',
      'Deit-Tiny R': 'darkviolet',
      'Deit-Base PT': 'black',}

cp1 = ['darkred','black','darkblue','darkorange','darkviolet','darkgreen']

cp2 = ["#ee4035","#f37736","#fdf498","#7bc043","#0392cf"]
cp3 = ['#4b3832','#854442','#fff4e6','#3c2f2f','#be9b7b']

key_to_plot = 'validation/val_0/lm_loss' #pile
key_to_plot = 'validation/val_1/lm_loss' #RP


def parse_job_string(job_string):
    # Regular expression pattern for each parameter
    patterns = {
        "WU": r"wu-([\d.]+)",
        "MaxLR": r"mxlr-([\d.e-]+)",
        "MinLR": r"mnlr-([\d.e-]+)"
    }

    result = ""
    for key in patterns:
        match = re.search(patterns[key], job_string)
        if match:
            if key == "MaxLR":
                # Convert to float and format as scientific notation
                value = "{:e}".format(float(match.group(1)))
                idx = len(value.split('e')[0])
                value = value[:idx].rstrip('0').rstrip('.') + value[idx:]
            else:
                value = match.group(1)
            result += f"{key}={value} "
            
    if 'pile' in job_string:
        result += 'Pile'
    else:
        result += 'RP'
        
    if '27000' in job_string:
        result += '27000'
    elif '10000' in job_string:
        result += '10000'
        
    if 'scratch' in job_string or 'none' in job_string :
        result += ' scratch'

    return result.rstrip()  # remove trailing space

def parse_job_string(job_string):
    # Regular expression pattern for each parameter
    patterns = {
        "Model": r"JOB-\d+_([\d.-]+B)_",    # Look for the model size format e.g., 7.1B
        "It": r"it-(\d+)",        # Look for the iterations number
        "WU": r"wu-([\d.]+)",
        "MaxLR": r"mxlr-([\d.e-]+)",
        "MinLR": r"mnlr-([\d.e-]+)",
        "Sch": r"sch-([\w]+)_",   # Adjusted the regex
        "Tr": r"tr-([\w-]+)_"  # Extract training dataset
    }

    result = ""
    for key in patterns:
        match = re.search(patterns[key], job_string)
        if match:
            if key == "MaxLR":
                # Convert to float and format as scientific notation
                value = "{:e}".format(float(match.group(1)))
                idx = len(value.split('e')[0])
                value = value[:idx].rstrip('0').rstrip('.') + value[idx:]
            else:
                value = match.group(1)
            result += f"{key}={value} "
            
    result += job_string.split("_")[-1]
    return result.rstrip()  # remove trailing space


def get_caption(key,use=dict(mx=True,wu=True,mn=True,ds=True,it=True)):
    temp = key.split(' ')
    out = ''
    
    if key in ['WU=0.0 MaxLR=3e-05 MinLR=3e-05 RP','WU=0.0 MaxLR=3e-05 MinLR=3e-05 Pile']:
        out = 'Constant 3e-05'
        if use['it']:
            if temp[3].endswith('10000'):
                out += "Iter. 10000"
            elif temp[3].endswith('27000'):
                out += "Iter. 27000"
            elif 'scratch' in key:
                out += "Iter. 0"
            else:
                out += "Iter. 143000"
        
        return out
    
    if use['wu']:
        out += temp[0].replace("="," ") + " "
        
    if use['mx']:
        out += temp[1].replace("="," ") + " "
    
    if use['mn']:
        out += temp[2].replace("="," ") + " "
        
    if use['ds']:
        if "RP" in temp[3]:
            out += 'RP' + " "
        else: 
            out += 'Pile' + " "
        
    if use['it']:
        if temp[3].endswith('10000'):
            out += "Iter. 10000"
        elif temp[3].endswith('27000'):
            out += "Iter. 27000"
        elif 'scratch' in key:
            out += "Iter. 0"
        else:
            out += "Iter. 143000"
            
    
            
        
    return out.strip(' ')


# Function to load tensorboard logs from one directory
def load_tensorboard_logs(subdir, parent_dir):
    # Create the full path to the subdirectory
    subdirectory_path = os.path.join(parent_dir, subdir)
    start_time = timeit.default_timer()
    
    # Load TensorBoard logs
    event_acc = EventAccumulator(subdirectory_path)
    event_acc.Reload()

    # Extract scalar values
    scalar_tags = event_acc.Tags()["scalars"]
    scalar_values = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        values = [(event.step, event.value) for event in events]
        scalar_values[tag] = values

    end_time = timeit.default_timer()
    iteration_time = end_time - start_time
    return (subdir, scalar_values, iteration_time)



def plot_single(data_dict,
                filter_=['Pile','MaxLR=1.5e-04','MaxLR=6e-04','MaxLR=3e-05',
                         '27000','10000'],
                upper_limit=5000,
                key_to_plot='validation/val_0/lm_loss',
                ylabel='Tokens (B)',
                xlabel='Pile Val Loss',
                savepath=None, #"lr3e-4-Pile-val-loss-warmup-50B.pdf",
                legend_kwargs=dict(fontsize=12),
                ylim=(2.18,2.55),
                figsize=(7.5,5),
                warmup_end={'0.01':True,'0.02':True,'0.005':True},
                savedir='esfomows/',
                color_dict={},
                ls_dict={},
                circles=False,
                default_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", ],
                default_ls =['-' for x in range(100)],
                label_dict={}
               ):  
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ii=0
    for i,(k,v) in enumerate(data_dict.items()):
        if len(v) == 0 \
           or reduce(np.logical_or,[x in k for x in filter_]):
            continue #skip
            
        temp = np.array(v[key_to_plot])
        
        iters_ = step_to_token(temp[:,0])
        value_ = temp[:,1]
        w = np.where(iters_ < upper_limit)

        ax.plot(iters_[w[0]],
                value_[w[0]],
                label=label_dict.get(k,k),
                linewidth=0.9,
                color=color_dict.get(k,default_colors[ii]),
                linestyle=ls_dict.get(k,default_ls[ii]))
        
        ax.set_xlabel(ylabel,)
        ax.set_ylabel(xlabel,)
    
        ii+=1
        
    vertical_bar = []
    for i,(k,v) in enumerate(data_dict.items()):  
        if len(v) == 0 or reduce(np.logical_or,[x in k for x in filter_]):
            continue
        v0 = np.array(v[key_to_plot]) #red pajama
        w = np.where(np.logical_and(1818 < v0[:,0],
                                    v0[:,0] < 1826,))
        
#         print(v0[w,0],v0[w,1])
        vertical_bar.append(step_to_token(v0[w,0]))

    # 662, 1318, 2633
    if warmup_end['0.005']:
        ax.axvline(x=step_to_token(662), color='lightgray', linestyle='-.',linewidth=0.5)
    if warmup_end['0.01']:
        ax.axvline(x=step_to_token(1318), color='lightgray', linestyle='-.',linewidth=0.5)
    if warmup_end['0.02']:
        ax.axvline(x=step_to_token(2633), color='lightgray', linestyle='-.',linewidth=0.5)
        
    if circles:
        x = [10,24,48]
        y = [2.58,2.5,2.445]
        plt.plot(x, y, 
                 marker='o', 
                 color='red', 
                 fillstyle='none', 
                 linestyle='None',
                 markersize=25,
                linewidth=15)  # Hollow red circles
        
#     for spine in ['left', 'right', 'bottom', 'top']:
#         ax.spines[spine].set_color('lightgrey')

    ax.legend(**legend_kwargs)
    ax.set_ylim(*ylim)
    ax.grid(True)
    if savepath:
        print("saving "+os.path.join(savedir,savepath))
        plt.savefig(os.path.join(savedir,savepath),bbox_inches='tight')
        savepath = savepath[:-4] + '.png'
        plt.savefig(os.path.join(savedir,savepath),bbox_inches='tight')
        # plt.savefig(savedir+savepath,bbox_inches='tight')
    plt.show()
    
def merge_tb_logs(list_dict):
        for k,v in list_dict.items():
            if len(v) > 1:
                temp = sorted(list(v.keys()),key=lambda x:int(x[4:11]))
                accum = {kk:dict(vv) for kk,vv in v[temp[0]].items()}


                for x in temp[1:]:
                    for kk,vv in accum.items():
                        try:
                            accum[kk].update(dict(v[x][kk]))
                        except KeyError as e:
                            print(e)
                            print("Missing key:",kk)
                            print("Outer key:",k)
                            

                #sort the zipped object
                list_dict[k] = {kk:sorted([x for x in zip(tmp.keys(),tmp.values())], key=lambda x: x[0]) for kk,tmp in accum.items()}
            else:
                list_dict[k] = list(v.values())[0]

        return list_dict    
    
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action='store_true', default=False, required=False)
    parser.add_argument("--tb-log-dir", '-t', type=str, required=True)
    parser.add_argument("--savedir", '-s', default="/gpfs/alpine/csc499/proj-shared/p2_continued_pretraining/plots",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    # Specify the parent directory containing directories with logs
    parent_dir = args.tb_log_dir
    
    if args.copy:
        os.system("rsync -r {} /gpfs/alpine/csc499/proj-shared/p2_continued_pretraining/tensorboard_iclr")

    # Get a list of subdirectories in the parent directory
    subdirectories = [subdir for subdir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, subdir))]

    start_time = timeit.default_timer()
    # Use a multiprocessing pool to load tensorboard logs in parallel
    with multiprocessing.Pool() as pool:
        total_time = 0
        result_dict = {}
        results = pool.starmap(load_tensorboard_logs, [(subdir, parent_dir) for subdir in subdirectories])
        for subdir, scalar_values, iteration_time in results:
            result_dict[subdir] = scalar_values
            total_time += iteration_time
            print(f'Loaded {os.path.join(parent_dir, subdir)} in {iteration_time} seconds')

    end_time = timeit.default_timer()
    print("All loaded in {} Seconds".format(end_time - start_time))
    
    # remove empty
    for x in [k for k,v in result_dict.items() if v == {}]:
        print(x)
        result_dict.pop(x)
    
    #accumulate
    d = {}
    for k,v in result_dict.items():
        try:
            d[parse_job_string(k)][k] = v
        except KeyError:
            d[parse_job_string(k)] = {k:v}

    rd_clone = merge_tb_logs(copy.deepcopy(d))
    key = 'validation/val_1/lm_loss'
    for k,v in rd_clone.items():
        plot_single({k:v},
                filter_=['to prevent error but not filtering'],
                # upper_limit=300,#step_to_token(105837.0),#300,#step_to_token(105837.0), 
                key_to_plot=key,
                ylabel='Tokens (B)',
                xlabel=key, 
                savepath=k.replace(" ","_") + ".pdf",
                savedir=args.savedir,
                legend_kwargs=dict(fontsize=12),
                ylim=[1.5,4],#(2.44,2.92),
                figsize=(9,5),
                warmup_end={'0.01':False,'0.02':False,'0.005':False},
               default_colors=cp1,
                label_dict={x:get_caption(x,use=dict(mx=True,wu=False,mn=False,ds=False,it=True)) \
                            for x in rd_clone.keys()})
