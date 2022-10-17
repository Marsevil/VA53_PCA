import os
import psutil
import re
import time
from PIL import Image, UnidentifiedImageError
from recognition import Recognition
from recognition_pca import PcaRecognition
from recognition_tensorflow import TensorflowRecognition
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

recog = PcaRecognition(0.8)
#recog = TensorflowRecognition(30)
start_training_time = time.time()
recog.train(train_image_list)
end_training_time = time.time()
process = psutil.Process(os.getpid())
time_training_lapsed = end_training_time - start_training_time

# Test with all image
score = 0
total = 0
total_testing_time = 0
dir_names = list(filter(HIDDEN_FILE_TESTER.match, os.listdir(TEST_DIR)))
for dir_name in dir_names:
    dir_path = os.path.join(TEST_DIR, dir_name)
    file_names = list(filter(HIDDEN_FILE_TESTER.match, os.listdir(dir_path)))
    for file_name in file_names:
        img_path = os.path.join(dir_path, file_name)
        img = None
        try:
            img = Recognition._load_image(img_path)
        except (FileNotFoundError, UnidentifiedImageError):
            print("Unable to read image: %s" % (img_path))
            continue

        start_time = time.time()
        found_label = recog.find(img)
        end_time = time.time()
        total_testing_time += end_time - start_time

        if found_label == dir_name:
            score += 1
        else:
            print("Wrong label : %s expected, %s found" % (dir_name, found_label))

        total += 1

print("Ratio : %i / %i = %f" % (score, total, score/total))
testing_time = total_testing_time / total
print("Training memory print : %f MB" % (process.memory_info().rss / 1024 ** 2))
print("Training time : %f sec" % (time_training_lapsed))
print("Average testing time per image : %f sec" % (testing_time))
print("OK")
