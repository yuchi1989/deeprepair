import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_feature_data', type=str, help='testing feature npy file')
    args = parser.parse_args()
    feature_data = np.load(args.testing_feature_data)
    object_list = feature_data["object_list"]
    person_class = 0
    for i in range(len(object_list)):
        if object_list[i] == "person":
            person_class = i
            break
    type2confusion = feature_data["confusion"]
    for i in range(len(object_list)):
        if i != person_class:
            print((person_class, i))
            print(type2confusion[(person_class, i)])


if __name__ == '__main__':
    main()
