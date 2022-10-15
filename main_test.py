import os
import re
from PIL import Image, UnidentifiedImageError
from recognition_pca import PcaRecognition
import numpy as np

""" CONSTANTS """
HIDDEN_FILE_TESTER = re.compile("^[^\.]")
TRAIN_DIR = "DataSet/Train"
TEST_DIR = "DataSet/Test"
DIR_NAMES = list(filter(HIDDEN_FILE_TESTER.match, os.listdir(TRAIN_DIR)))


""" MAIN """
# Read all files
train_image_list = {}
for dir_name in DIR_NAMES:
    directory_path = os.path.join(TRAIN_DIR, dir_name)
    file_names = list(
            filter(
                HIDDEN_FILE_TESTER.match,
                os.listdir(directory_path)
                )
            )
    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        train_image_list[file_path] = dir_name

recog = PcaRecognition()
recog.train(train_image_list)

# Test with all image
score = 0
total = 0
dir_names = list(filter(HIDDEN_FILE_TESTER.match, os.listdir(TEST_DIR)))
for dir_name in dir_names:
    dir_path = os.path.join(TEST_DIR, dir_name)
    file_names = list(filter(HIDDEN_FILE_TESTER.match, os.listdir(dir_path)))
    for file_name in file_names:
        img_path = os.path.join(dir_path, file_name)
        img = None
        try:
            img = Image.open(img_path).convert("L")
        except (FileNotFoundError, UnidentifiedImageError):
            print("Unable to read image: %s" % (img_path))
            continue

        img = np.asarray(img)
        found_label = recog.find(img)

        if found_label == dir_name:
            score += 1
        else:
            print("Wrong label : %s expected, %s found" % (dir_name, found_label))

        total += 1

print("Ratio : %i / %i = %f" % (score, total, score/total))

print("OK")

