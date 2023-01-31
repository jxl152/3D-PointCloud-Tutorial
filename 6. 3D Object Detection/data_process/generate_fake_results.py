import os
import sys
import random


gt_dir = "../data/validation_label/"
result_dir = "../data/fake_label/data/"

if __name__ == "__main__":
    random.seed(42)

    fake_result = os.listdir(result_dir)
    if fake_result:
        print("fake_label has already had data.")
        sys.exit(0)

    for file_name in os.listdir(gt_dir):
        with open(gt_dir+file_name, 'r') as gt_file:
            with open(result_dir+file_name, 'w') as fr_file:
                # iterate over the labeled objects
                for line in gt_file.readlines():
                    values = line.rstrip().split()
                    type = values[0]
                    if type in ("Car", "Pedestrian", "Cyclist"):
                        # randomly modify its dimensions and locations
                        # dimensions = values[-7:-4]
                        # locations = values[-4:-1]
                        # if random.random() > 0.8:
                        #     height, width, length = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
                        #     variant = random.uniform(0.8, 1.2)
                        #     values[-7:4] = [str(height * variant), str(width * variant), str(length * variant)]
                        # if random.random() > 0.9:
                        #     x, y, z = float(locations[0]), float(locations[1]), float(locations[2])
                        #     variant = random.uniform(0.9, 1.1)
                        #     values[-4:-1] = [str(x * variant), str(y * variant), str(z * variant)]
                        # fake a score
                        score = random.random()
                        values.append(str(score))
                        fr_file.write(" ".join(values)+"\n")
