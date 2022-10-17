from PIL import Image
import numpy as np


class Recognition:

    IMG_SIZE = (512, 512)

    def _load_image(img_path: str) -> np.array:
        img = Image.open(img_path)\
                .convert("L")\
                .resize(Recognition.IMG_SIZE)

        return np.asarray(img)

    def train(self, img_list: dict):
        """
        Train the model with a database of label / image

        Arguments:
            imgList (dict<str, str>): A dictionnary of label / path. Label is
            used to identify a face and path to load an associated image
        """

        pass

    def find(self, img: np.matrix) -> str:
        """
        Compare an image to the images of the model

        Arguments:
            img: a numpy matrix containing image data
        """

        pass

