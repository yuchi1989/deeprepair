import numpy as np

def top_confusions(epoch_confusion_file, n = 5):
    information = np.load(epoch_confusion_file, allow_pickle=True)
    epoch = []
    nature_accuracy = []
    dogcat_natural_confusion = []
    top_natural_confusion = []
    lr = []
    for i in information:
        epoch.append(i["epoch"])
        lr.append(i["lr"])
        nature_accuracy.append(i["accuracy"]/100)
        dogcat_natural_confusion.append(i["confusion"][(5, 3)])
        top_natural_confusion.append(i["confusion"][sorted(i["confusion"], key=i["confusion"].get, reverse=True)[0]])
    print("first stage")
    max_acc = 0
    max_index = 0
    for i in range(len(nature_accuracy)):
        if nature_accuracy[i] >= max_acc:
            max_acc = nature_accuracy[i]
            max_index = i

    print(nature_accuracy[max_index])
    print(information[max_index]["confusion"][(5, 3)])
    print(information[max_index]["confusion"][(3, 5)])
    confusion_matrix = information[max_index]["confusion"]
    avg_pair_confusion = {}
    for i in range(10):
        for j in range(i + 1, 10):
            avg_pair_confusion[(i,j)] = (confusion_matrix[(i, j)] + confusion_matrix[(j, i)])/2
    keys = sorted(avg_pair_confusion, key=avg_pair_confusion.get, reverse=True)[:n]
    for i in range(n):
        print("")
        print(str(keys[i]) + " avg: " + str(avg_pair_confusion[keys[i]]))
        print(str(keys[i]) + ": " + str(confusion_matrix[keys[i]]))
        print(str(keys[i][::-1]) + ": " + str(confusion_matrix[keys[i][::-1]]))

    #print(top_natural_confusion[max_index])

def draw_graph():

    information = np.load("./log/cifar10_resnet_2_4_epoch_confusion.npy", allow_pickle=True)
    nature_accuracy = []
    dogcat_confusion = []
    autotruck_confusion = []
    airship_confusion = []
    cur_acc = 0
    for i in information:
        acc = i["accuracy"]/100
        if acc > cur_acc:
            nature_accuracy.append(acc)
            cur_acc = acc
            dogcat_confusion.append((i["confusion"][(5, 3)] + i["confusion"][(3, 5)])/2)
            autotruck_confusion.append((i["confusion"][(1, 9)] + i["confusion"][(9, 1)])/2)
            airship_confusion.append((i["confusion"][(0, 8)] + i["confusion"][(8, 0)])/2)
    print("first stage")


    print(nature_accuracy[-1])
    print(information[-1]["confusion"][(5,3)])
    print(information[-1]["confusion"][(3,5)])

    print(nature_accuracy[-2])
    print(dogcat_confusion[-2])
    print(autotruck_confusion[-2])
    print(airship_confusion[-2])
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    ax.plot(nature_accuracy, dogcat_confusion, 'r--', label="dog/cat confusion")
    ax.plot(nature_accuracy, autotruck_confusion, 'g-', label="car/truck confusion")
    ax.plot(nature_accuracy, airship_confusion, 'b-', label="airplane/ship")
    plt.ylabel("confusion")
    plt.xlabel("natural accuracy")
    legend = ax.legend(loc='lower left', shadow=True, fontsize=14)
    plt.savefig("cifar10.pdf", bbox_inches='tight')
    plt.show()


def draw_bias_graph(pretrain, repair, second, third):
    information = np.load(pretrain, allow_pickle=True)
    information2 = np.load(repair, allow_pickle=True)
    nature_accuracy = []
    bias = []
    extra_info = []
    loss = []
    for i in information:
        acc = i["accuracy"]/100
        nature_accuracy.append(acc)
        loss.append(0)
        bias.append(compute_pairwise_bias(i["confusion"], second, third))
    
    for i in information2:
        acc = i["accuracy"]/100
        nature_accuracy.append(acc)
        loss.append(i["loss"]/10+1)
        bias.append(compute_pairwise_bias(i["confusion"], second, third))
    print(bias)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    x = range(len(nature_accuracy))
    ax.plot(x, nature_accuracy, 'r-', label="accuracy")
    ax.plot(x, loss, 'b-', label="loss")
    ax.plot(x, bias, 'g-', label="airplane, cat bias")
    #ax.plot(x, extra_info, 'b-', label="dog->cat confusion")
    plt.ylabel("accuracy/bias")
    plt.xlabel("epoch")
    legend = ax.legend(loc='best', shadow=True, fontsize=14)
    plt.savefig("cifar10_twophase_pairwise.pdf", bbox_inches='tight')
    plt.show()


def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]
    
    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))


def compute_pairwise_bias(confusion_matrix, second, third):
    object_list = []
    for i in range(10):
        object_list.append(i)
    bias = 0
    for o in object_list:
        if second == o or third == o:
            continue
        bias += compute_bias(confusion_matrix, o, second, third)
    return bias

def top_bias(epoch_confusion_file, n = 5):
    information = np.load(epoch_confusion_file, allow_pickle=True)
    epoch = []
    nature_accuracy = []
    dogcat_natural_confusion = []
    top_natural_confusion = []
    avg_pair_confusion = {}
    lr = []
    for i in information:
        epoch.append(i["epoch"])
        lr.append(i["lr"])
        nature_accuracy.append(i["accuracy"]/100)
        dogcat_natural_confusion.append(i["confusion"][(5, 3)])
        top_natural_confusion.append(i["confusion"][sorted(i["confusion"], key=i["confusion"].get, reverse=True)[0]])
    print("first stage")
    max_acc = 0
    max_index = 0
    for i in range(len(nature_accuracy)):
        if nature_accuracy[i] >= max_acc:
            max_acc = nature_accuracy[i]
            max_index = i

    print(nature_accuracy[max_index])
    print(information[max_index]["confusion"][(5, 3)])
    print(information[max_index]["confusion"][(3, 5)])
    confusion_matrix = information[max_index]["confusion"]

    for i in range(10):
        for j in range(i + 1, 10):
            avg_pair_confusion[(i,j)] = (confusion_matrix[(i, j)] + confusion_matrix[(j, i)])/2
            avg_pair_confusion[(j,i)] = (confusion_matrix[(i, j)] + confusion_matrix[(j, i)])/2

    triplet_bias = {}
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i == j or j == k or i ==k:
                    continue
                triplet_bias[(i, j, k)] = abs(avg_pair_confusion[(i, j)] - avg_pair_confusion[(i, k)])

    pairwise_bias = {}
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            pairwise_bias[(i, j)] = 0
            for k in range(10):
                if j == k or i ==k:
                    continue
                pairwise_bias[(i, j)] += triplet_bias[(k, i, j)]

    keys = sorted(pairwise_bias, key=pairwise_bias.get, reverse=True)[:n]
    for i in range(n):
        print("")
        print(str(keys[i]) + " triplet: " + str(pairwise_bias[keys[i]]))
        #print(str((keys[i][0],keys[i][1])) + ": " + str(avg_pair_confusion[(keys[i][0], keys[i][1])]))
        #print(str((keys[i][0],keys[i][2])) + ": " + str(avg_pair_confusion[(keys[i][0], keys[i][2])]))

    #print(top_natural_confusion[max_index])
    print(pairwise_bias[(1,3)])

if __name__ == '__main__':
    #top_confusions("./log/cifar10_resnet_2_4_epoch_confusion.npy", 3)
    #draw_graph()
    #draw_dog_cat_0_confusion("./log/cifar10_resnet_2_4_epoch_confusion.npy", "./log/cifar10_resnet_2_4_dogcat_2_epoch_confusion6.npy")
    #top_bias("../../exp_2/log/cifar10_resnet_2_4_epoch_confusion.npy", 10)
    #draw_bias_graph("../../exp_2/log/cifar10_resnet_2_4_epoch_confusion.npy", "./log/cifar10_resnet_2_4_bias_351_epoch_confusion_5.npy")
    draw_bias_graph("../../exp_2/log/cifar10_resnet_2_4_epoch_confusion.npy", "./log/epoch_confusion_2.npy", 1, 3)
    #draw_bias_graph("../../exp_2/log/cifar10_resnet_2_4_epoch_confusion.npy", "./log/epoch_confusion_4.npy", 3, 1)