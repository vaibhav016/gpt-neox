import os 

d = os.listdir('tensorboard')
tf_recs = [x for x in d if 'tfevents' in x]
runs = list(set([x.split('.')[3] for x in tf_recs]))

for run in runs:
    if not os.path.exists(f'tensorboard/run_{run}'):
        os.makedirs(f'tensorboard/run_{run}')
    command = f'mv tensorboard/events.out.tfevents.{run}* tensorboard/run_{run}'
    print(command)
    os.system(command)

