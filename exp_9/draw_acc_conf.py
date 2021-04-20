from matplotlib import pyplot as plt
coco_acc_conf_list = [
('w-os 0.12', 66.03, 0.005, 0.342),
('w-os 0.2', 66.02, 0.004, 0.236),
('w-os 0.3', 66.01, 0.002, 0.142),
('w-os 0.4', 65.99, 0.0002, 0.031),

('w-bn 0.2 retrain', 66.03, 0.002, 0.315),
('w-bn 0.4 retrain', 65.94, 0.0002, 0.234),
('w-bn 0.6 retrain', 65.79, 0, 0.123),
('w-bn 0.8 retrain', 65.60, 0, 0.049),

('orig', 66.04, 0.007, 0.469)
]


cifar10_acc_conf_list = [
('w-os 0.2', 87.44, 0.072, 0.092),
('w-os 0.3', 86.68, 0.054, 0.073),
('w-os 0.4', 85.25, 0.03, 0.044),
('w-os 0.49', 80.55, 0.006, 0.005),
('w-os 0.5', 72.88, 0.0, 0.0),

('w-bn 0.4', 82.34, 0.023, 0.078),
('w-bn 0.5', 79.62, 0.016, 0.067),
('w-bn 0.5 retrain', 80.83, 0.021, 0.061),
('w-bn 1.0 retrain', 67.38, 0.0, 0.005),

('orig', 87.47, 0.085, 0.107)
]

color_map = {'orig': 'black', 'w-os': 'red', 'w-bn': 'blue'}

# 'coco', 'cifar10'
dataset = 'cifar10'

if dataset == 'cifar10':
    acc_conf_list = cifar10_acc_conf_list
elif dataset == 'coco':
    acc_conf_list = coco_acc_conf_list



for method, acc, conf1, conf2 in acc_conf_list:
    conf = conf1+conf2

    color = 'yellow'
    for m, color in color_map.items():
        if m in method:
            color = color_map[m]
            break
    print(color)
    plt.scatter(acc, conf*100, marker='o', color=color, label=method)
    plt.xlabel('accuracy(%)', fontsize=20)
    plt.ylabel('confusion(%)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(dataset, fontsize=25)

plt.legend(prop={'size': 12})
plt.savefig('acc_conf_plot_'+dataset+'.pdf')
