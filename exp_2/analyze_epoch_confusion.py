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


def draw_dog_cat_0_confusion(pretrain, repair):
    information = np.load(pretrain, allow_pickle=True)
    information2 = np.load(repair, allow_pickle=True)
    nature_accuracy = []
    dogcat_confusion = []

    for i in information:
        acc = i["accuracy"]/100
        nature_accuracy.append(acc)
        dogcat_confusion.append((i["confusion"][(5, 3)] + i["confusion"][(3, 5)])/2)
    
    for i in information2:
        acc = i["accuracy"]/100
        nature_accuracy.append(acc)
        dogcat_confusion.append((i["confusion"][(5, 3)] + i["confusion"][(3, 5)])/2)
            
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    x = range(len(nature_accuracy))
    ax.plot(x, nature_accuracy, 'r-', label="accuracy")
    ax.plot(x, dogcat_confusion, 'g-', label="dog/cat confusion")
    plt.ylabel("accuracy/confusion")
    plt.xlabel("epoch")
    legend = ax.legend(loc='best', shadow=True, fontsize=14)
    plt.savefig("cifar10_twophase6.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #top_confusions("./log/cifar10_resnet_2_4_epoch_confusion.npy", 3)
    #draw_graph()
    draw_dog_cat_0_confusion("./log/cifar10_resnet_2_4_epoch_confusion.npy", "./log/cifar10_resnet_2_4_dogcat_2_epoch_confusion6.npy")