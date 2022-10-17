import numpy as np
from PIL import UnidentifiedImageError
from sklearn.decomposition import PCA
from recognition import Recognition


class PcaRecognition(Recognition):
    def __init__(self, keep_data_ratio: float) -> None:
        assert keep_data_ratio >= 0 and keep_data_ratio <= 1

        self._eigenfaces = None
        self._weights = None
        self._mean = None
        self._labels = None
        self._keep_data_ratio = keep_data_ratio

    def train(self, imgList: dict):
        files = list(imgList.keys())
        self._labels = list(imgList.values())

        facematrix = []

        for file in files:
            img = None
            try:
                img = Recognition._load_image(file)
            except (FileNotFoundError, UnidentifiedImageError):
                print(file, "Can't be opened as an image")
                # TODO: Drop associated key
                continue

            img = img.flatten()
            facematrix.append(img)

        facematrix = np.array(facematrix)
        pca = PCA().fit(facematrix)

        # Determine which vectors should be kept according
        # to the coeff parameter
        cumulated_variance_ratio =\
            np.cumsum(pca.explained_variance_ratio_)
        conditions = cumulated_variance_ratio >= self._keep_data_ratio
        # Keep only relevent vectors
        if conditions.any():
            last_idx_to_keep = np.where(conditions)[0][0]+1
            self._eigenfaces = pca.components_[:last_idx_to_keep]
        else:
            self._eigenfaces = pca.components_

        self._mean = pca.mean_
        self._weights = self._eigenfaces @ (facematrix - self._mean).T

    def find(self, img: np.matrix) -> str:
        assert (self._eigenfaces is not None
                and self._mean is not None
                and self._labels is not None
                and self._weights is not None)

        flattened_img = img.reshape(1, -1)

        img_weight = self._eigenfaces @\
            (flattened_img - self._mean).T
        del flattened_img

        euclidian_distances = np.linalg.norm(
                self._weights - img_weight,
                axis=0)

        best_match = np.argmin(euclidian_distances)

        return self._labels[best_match]
