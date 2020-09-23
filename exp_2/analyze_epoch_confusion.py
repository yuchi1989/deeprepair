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

    information = np.load("./log/cifar10_resnet_2_2_epoch_confusion.npy", allow_pickle=True)
    information2 = np.load("/home/yuchi/CutMix-PyTorch/global_epoch_confusion/global_epoch_confusion_cutmix_90_pscore_further_triplet_sym.npy", allow_pickle=True)
    epoch = []
    nature_accuracy = []
    dogcat_natural_confusion = []
    top_natural_confusion = []
    lr = []
    for i in information:
        epoch.append(i["epoch"])
        lr.append(i["lr"])
        nature_accuracy.append(i["accuracy"]/100)
        dogcat_natural_confusion.append(i["confusion"][(5,3)])
        top_natural_confusion.append(i["confusion"][sorted(i["confusion"], key=i["confusion"].get, reverse=True)[0]])
    print("first stage")
    max_acc = 0
    max_index = 0
    for i in range(len(nature_accuracy)):
        if nature_accuracy[i] >= max_acc:
            max_acc = nature_accuracy[i]
            max_index = i

    print(nature_accuracy[max_index])
    print(information[max_index]["confusion"][(5,3)])
    print(information[max_index]["confusion"][(3,5)])
    #print(sorted(information[max_index]["confusion"], key=information[max_index]["confusion"].get, reverse=True))
    print(top_natural_confusion[max_index])
    for i in information2:
        epoch.append(i["epoch"] + len(information))
        lr.append(i["lr"])
        nature_accuracy.append(i["accuracy"]/100)
        dogcat_natural_confusion.append(i["confusion"][(5,3)])
        top_confusion = 0
        top_natural_confusion.append(i["confusion"][sorted(i["confusion"], key=i["confusion"].get, reverse=True)[0]])

    print("second stage")
    max_acc = 0
    max_index = 0
    for i in range(len(information), len(nature_accuracy)):
        if nature_accuracy[i] >= max_acc:
            max_acc = nature_accuracy[i]
            max_index = i

    print(nature_accuracy[max_index])
    print(information2[max_index - len(information)]["confusion"][(5,3)])
    print(information2[max_index - len(information)]["confusion"][(3,5)])
    print(top_natural_confusion[max_index])
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    ax.plot(epoch, dogcat_natural_confusion, 'r--', label="dog->cat natural confusion")
    ax.plot(epoch, nature_accuracy, 'g-', label="test data natural accuracy")
    ax.plot(epoch, lr, 'b-', label="learning rate")
    ax.plot(epoch, top_natural_confusion, color='gray', linestyle='dashdot', label="largest pair natural confusion")
    plt.ylabel("accuracy/confusion")
    plt.xlabel("training epochs")
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    plt.show()


if __name__ == '__main__':
    top_confusions("./log/cifar10_resnet_2_2_epoch_confusion.npy", 3)
    #draw_graph()