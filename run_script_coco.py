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
    "filename": "exp_weighted_loss",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 0,
},
'w-bn': {
    "param_name": "ratio",
    "filename": "bn",
    "replace": " --replace",
    # "batch_size": 64,
    # "extra_batch_size": 64,
},
'w-loss': {
    "param_name": "target_weight",
    "filename": "exp_weighted_loss",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 0,
},
'w-dbr': {
    "param_name": "lam",
    "filename": "dbr",
    "replace": "",
    # "batch_size": 64,
    # "extra_batch_size": 6,
}
}



dataset_model_classes = [('coco', 'coco', ("bus", "person", "clock")), ('coco_gender', 'coco_gender', ("handbag", "woman", "man"))]
tasks = ['confusion', 'bias']
methods = ['w-aug', 'w-bn', 'w-loss', 'w-dbr']
params = [0.1, 0.3, 0.5, 0.7, 0.9]
epochs = 6
verbose = ""

dataset_model_classes = [('coco', 'coco', ("bus", "person", "clock")), ('coco_gender', 'coco_gender', ("handbag", "woman", "man"))]
tasks = ['confusion']
methods = ['w-aug']
params = [0.1]
epochs = 1


with open('tmp_log_coco.txt', 'w') as f_out:
    pass
t0 = time.time()
for dataset, model, classes in dataset_model_classes:
    first, second, third = classes

    if model == 'coco':
        model_path = 'models/coco_original_model/model_best.pth.tar'
        class_num = 80
    elif model == 'cocogender':
        model_path = 'models/cocogender_original_model/model_best.pth.tar'
        class_num = 81

    for task in tasks:
        for method in methods:
            for param in params:
                param_name = method_properties[method]["param_name"]
                filename = method_properties[method]["filename"]
                replace = method_properties[method]["replace"]

                filepath = os.path.join('exp_9', dataset, 'confusion_and_bias', 'repair_'+task+'_'+filename+'.py')
                expname = 'runs/'+dataset+'_'+task+'_'+method+'_'+str(param)

                cmd = f"python2 {filepath} --pretrained {model_path} --log_dir {expname} --first {first} --second {second} --ann_dir '../coco/annotations' --image_dir '../coco/' --class_num {class_num} --{param_name} {param}"+replace

                if task == 'bias':
                    cmd += ' --third '+str(third)
                print('-'*20)
                print(cmd)
                print('-'*20)
                with open('tmp_log.txt', 'a') as f_out:
                    f_out.write(cmd+'\n')
                    f_out.write(str(time.time()-t0)+'\n')
                execute(cmd)
