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

dataset_model_classes = [('cifar10', 'resnet-18', (3, 5, 2))]
tasks = ['confusion']
methods = ['w-aug']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
# epochs = 2


config_list = [
('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-aug', 0.7),
('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-bn', 0.9),
('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-loss', 0.9),
('cifar10', 'resnet-18', (3, 5, 2), 'confusion', 'w-dbr', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-aug', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-bn', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-loss', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'confusion', 'w-dbr', 0.7),
('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-aug', 0.7),
('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-bn', 0.9),
('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-loss', 0.7),
('cifar100', 'resnet-34', (98, 35, 11), 'confusion', 'w-dbr', 0.9),

('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-aug', 0.7),
('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-bn', 0.9),
('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-loss', 0.9),
('cifar10', 'resnet-18', (3, 5, 2), 'bias', 'w-dbr', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-aug', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-bn', 0.9),
('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-loss', 0.7),
('cifar10', 'vggbn-11', (3, 5, 2), 'bias', 'w-dbr', 0.9),
('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-aug', 0.7),
('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-bn', 0.9),
('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-loss', 0.9),
('cifar100', 'resnet-34', (98, 35, 11), 'bias', 'w-dbr', 0.9),
]



def execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num):
    if model == 'coco':
        model_path = 'models/coco_original_model/model_best.pth.tar'
        class_num = 80
    elif model == 'coco_gender':
        model_path = 'models/cocogender_original_model/model_best.pth.tar'
        class_num = 81
    else:
        raise
    first, second, third = classes

    param_name = method_properties[method]["param_name"]
    filename = method_properties[method]["filename"]
    replace = method_properties[method]["replace"]

    filepath = os.path.join('exp_9', dataset, 'confusion_and_bias', 'repair_'+task+'_'+filename+'.py')
    expdir = os.path.join('runs', dataset+'_'+task+'_'+str(first)+'_'+str(second)+'_'+str(third)+'_'+str(rep_num))
    if not os.path.isdir(expdir):
        os.mkdir(expdir)
    expname = os.path.join(expdir, dataset+'_'+task+'_'+method+'_'+str(param))

    cmd = f"python2 {filepath} --pretrained {model_path} --log_dir {expname} --first {first} --second {second} --ann_dir '../coco/annotations' --num_epochs {epochs} --image_dir '../coco/' --class_num {class_num} --{param_name} {param}"+replace

    if task == 'bias':
        cmd += ' --third '+str(third)
    print('-'*20)
    print(cmd)
    print('-'*20)
    with open(log_filename, 'a') as f_out:
        f_out.write(cmd+'\n')
        f_out.write(str(time.time()-t0)+'\n')
    execute(cmd)

if __name__ == '__main__':
    # ['grid', 'specific']
    mode = 'specific'
    t0 = time.time()
    if mode == 'grid':
        log_filename = 'tmp_log.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for dataset, model, classes in dataset_model_classes:
            for task in tasks:
                for method in methods:
                    for param in params:
                        execute_cmd(dataset, model, classes, task, method, param, log_filename, t0)
    elif mode == 'specific':
        rep_nums = 4
        log_filename = 'tmp_log_coco_specific.txt'
        with open(log_filename, 'w') as f_out:
            pass
        for rep_num in range(rep_nums):
            for dataset, model, classes, task, method, param in config_list:
                execute_cmd(dataset, model, classes, task, method, param, log_filename, t0, rep_num)
    else:
        raise
