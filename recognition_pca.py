import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA


class PcaRecognition:
    def __init__(self) -> None:
        self._pca = None
        self._weights = None
        self._labels = None

    def train(self, imgList: dict):
        files = list(imgList.keys())
        self._labels = list(imgList.values())

        facematrix = []

        for file in files:
            img = None
            try:
                img = Image.open(file).convert('L')
            except (FileNotFoundError, UnidentifiedImageError):
                print(file, "Can't be opened as an image")
                # TODO: Drop associated key
                continue

            img = np.asarray(img).flatten()
            facematrix.append(img)

        facematrix = np.array(facematrix)
        self._pca = PCA().fit(facematrix)

        eigenfaces = self._pca.components_

        self._weights = eigenfaces @ (facematrix - self._pca.mean_).T

    def find(self, img: np.matrix) -> str:
        assert (self._pca is not None
                and self._labels is not None
                and self._weights is not None)

        flattened_img = img.reshape(1, -1)

        img_weight = self._pca.components_ @ (flattened_img - self._pca.mean_).T
        del flattened_img

        euclidian_distances = np.linalg.norm(self._weights - img_weight, axis=0)

        best_match = np.argmin(euclidian_distances)

        return self._labels[best_match]
