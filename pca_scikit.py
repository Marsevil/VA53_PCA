import os
import sys
import re
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

FILTER = re.compile("^[^\.]")
BASE_DIR = r"DataSet"
TRAIN_PATH = os.path.join(BASE_DIR, r"Train")

files = os.listdir(TRAIN_PATH)
files = list(filter(FILTER.match, files))
facematrix = []
facelabel = []
for file in files:
    try:
        img = Image.open(os.path.join(TRAIN_PATH, file)).convert("L")
    except (FileNotFoundError, UnidentifiedImageError):
        print(file, "Can't be opened as an image")
        continue

    img = np.asarray(img).flatten()
    facematrix.append(img)
    facelabel.append(file)

facematrix = np.array(facematrix)

from sklearn.decomposition import PCA

pca = PCA().fit(facematrix)
eigenfaces = pca.components_

weights = eigenfaces @ (facematrix - pca.mean_).T

queryPath = sys.argv[1]
query = Image.open(queryPath).convert("L")
query = np.matrix(query).reshape(1, -1)
query_weight = eigenfaces @ (query - pca.mean_).T
euclidian_distances = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidian_distances)

print("Best match is %s with an euclidian distance value equal to %f" % (facelabel[best_match], euclidian_distances[best_match]))

