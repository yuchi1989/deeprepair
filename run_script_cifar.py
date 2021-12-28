import os
import time
from subprocess import PIPE, run

def execute(command):
    result = run(command, universal_newlines=True, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    print(result.stderr)


method_properties = {
'w-aug': {
    "param_name": "weight",
    "filename": "exp_oversampling",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 0,
},
'w-bn': {
    "param_name": "ratio",
    "filename": "exp_newbn",
    "replace": " --replace",
    "batch_size": 128,
    "extra_batch_size": 128,
},
'w-loss': {
    "param_name": "target_weight",
    "filename": "exp_weighted_loss",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 10,
},
'w-dbr': {
    "param_name": "lam",
    "filename": "dbr",
    "replace": "",
    "batch_size": 128,
    "extra_batch_size": 10,
}
}



dataset_model_classes = [('cifar10', 'resnet-18', (3, 5, 2)), ('cifar10', 'vggbn-11', (3, 5, 2)), ('cifar100', 'resnet-34', (98, 35, 11))]
tasks = ['confusion', 'bias']
methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
epochs = 60
verbose = ""

dataset_model_classes = [('cifar10', 'resnet-18', (3, 5, 2)), ('cifar100', 'resnet-34', (98, 35, 11))]
# tasks = ['confusion']
methods = ['w-loss', 'w-dbr', 'w-bn', 'w-aug']
# params = [0.1]
# epochs = 2


with open('tmp_log.txt', 'w') as f_out:
    pass
t0 = time.time()
for dataset, model, classes in dataset_model_classes:
    first, second, third = classes

    if model == 'vggbn-11':
        model_type = 'resnet'
        model_depth = '18'
        vggbn = '_vggbn'
    else:
        model_type, model_depth = model.split('-')
        vggbn = ''

    if model == 'vggbn-11':
        model_path = 'models/cifar10_vggbn_2_4/model_best.pth.tar'
    elif model == 'resnet-18':
        model_path = 'models/cifar10_resnet18_2_4/model_best.pth.tar'
    elif model == 'resnet-34':
        model_path = 'models/cifar100_resnet34/model_best.pth.tar'

    for task in tasks:
        for method in methods:
            for param in params:
                param_name = method_properties[method]["param_name"]
                filename = method_properties[method]["filename"]
                replace = method_properties[method]["replace"]
                batch_size = method_properties[method]["batch_size"]
                extra_batch_size = method_properties[method]["extra_batch_size"]

                filepath = os.path.join('exp_7', dataset, task, 'repair_'+task+'_'+filename+vggbn+'.py')
                expname = dataset+'_'+model+'_'+task+'_'+method+'_'+str(param)

                cmd = f"python3 {filepath} --net_type {model_type} --dataset {dataset} --depth {model_depth} --expname {expname} --epochs {epochs} --lr 0.1 --beta 1.0 --cutmix_prob 0 --pretrained {model_path} --batch_size {batch_size} --extra {extra_batch_size} --first {first} --second {second} --{param_name} {param}"+replace+verbose

                if task == 'bias':
                    cmd += ' --third '+str(third)
                print('-'*20)
                print(cmd)
                print('-'*20)
                with open('tmp_log.txt', 'a') as f_out:
                    f_out.write(cmd+'\n')
                    f_out.write(str(time.time()-t0)+'\n')
                execute(cmd)
