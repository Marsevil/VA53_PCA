import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from recognition import Recognition


class TensorflowRecognition(Recognition):
    """
    A class that implement face recognition througt tensorflow ML algorithms
    """

    def __init__(self):
        self._labels = []
        self._model = None
        self._probability_model = None

    def _prepare_data(self, img_list: dict) -> tuple:
        """
        Build a numeric labels index list with list of string labels

        Args:
            img_list (dict<str, str>): Dictionnary of path as keys & labels as
                values

        Returns:
            Return a tuple of (indexes, images_path)
        """

        labels_idx = []
        for _, label in img_list.items():
            label_idx = None
            if label in self._labels:
                label_idx = self._labels.index(label)
            else:
                self._labels.append(label)
                label_idx = len(self._labels) - 1

            labels_idx.append(label_idx)

        labels_idx = np.asarray(labels_idx)

        return (labels_idx, list(img_list.keys()))

    def _load_images(self, images_path: list) -> np.array:
        """
        Load images from a list of image path

        Arguments:
            images_path (list<str>): List of image path

        Returns:
            A numpy array of numpy array (image)
        """

        images = []
        for image_path in images_path:
            img = None
            try:
                img = Recognition._load_image(image_path)
            except(FileNotFoundError, UnidentifiedImageError):
                print("Unable to read image %s" % (image_path))
                continue

            img = np.asarray(img)
            images.append(img)

        return np.asarray(images)

    def train(self, img_list: dict):
        """
        Train the model with a database of label / image

        Arguments:
            imgList (dict<str, str>): A dictionnary of label / path. Label is
            used to identify a face and path to load an associated image
        """

        # Build training data
        train_labels_idx, train_images_path = self._prepare_data(img_list)
        train_images = self._load_images(train_images_path)

        train_images = train_images / 255

        # Build model
        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(
                input_shape=Recognition.IMG_SIZE),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(11)
            ])

        self._model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=["accuracy"]
                )

        self._model.fit(train_images, train_labels_idx, epochs=40)

        self._probability_model = tf.keras.Sequential([
            self._model,
            tf.keras.layers.Softmax()])

    def find(self, img: np.matrix) -> str:
        """
        Compare an image to the images of the model

        Arguments:
            img: a numpy matrix containing image data
        """

        assert self._model is not None and self._probability_model is not None

        prediction = self._probability_model.predict(np.expand_dims(img, 0))

        best_match = np.argmax(prediction)

        return self._labels[best_match]
