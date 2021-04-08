"""
Coco dataset for image captioning.

See Also: https://cocodataset.org/
"""
import bisect
import json
import os
from typing import List, Tuple, Callable

import numpy as np
import tensorflow as tf


def default_cache_dir() -> str:
    """
    Returns:
        Default cache directory for keras datasets.
    """
    return os.path.join(os.path.expanduser('~'), '.keras', 'datasets')


class CocoAnnotations:
    """Represents annotations file for coco dataset."""

    def __init__(self, cache_dir: str):
        """
        Arguments:
            cache_dir: Directory to save the annotation data file.
        """
        self.cache_dir = cache_dir
        self.annotation_file = os.path.join(self.cache_dir, 'annotations',
                                            'captions_train2014.json')
        self.annotations_origin_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

    def path(self) -> str:
        """
        Returns:
            Path to annotations data file.
        """
        return self.annotation_file

    def exists(self) -> bool:
        """
        Returns:
            True if the annotations file exists.
        """
        return os.path.exists(self.annotation_file)

    def download(self, remove_zip=True) -> str:
        """
        Downloads the compressed annotations file and extracts the contents.

        Arguments:
            remove_zip(bool): remove the compressed file after downloading
                and extracting is complete.

        Returns:
            Path to annotations file downloaded.
        """
        annotations_zip = tf.keras.utils.get_file(
            fname='captions.zip',
            origin=self.annotations_origin_url,
            extract=True)
        print(f'Downloaded annotations at: {annotations_zip}.')
        print(f'Extracted at: {self.path()}')

        # Clean up the zip file.
        if remove_zip:
            os.remove(annotations_zip)
        return self.annotation_file


class CocoImages:
    """
    Represents the image set of COCO public dataset.
    Provides main functionality of downloading the dataset.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.image_dir = os.path.join(self.cache_dir, 'train2014')
        self.images_origin_url = 'http://images.cocodataset.org/zips/train2014.zip'

    def path(self) -> str:
        """
        Returns:
            Path to images data directory.
        """
        return self.image_dir

    def exists(self) -> bool:
        """
        Returns:
            True if the images directory exists.
        """
        return os.path.exists(self.image_dir)

    def download(self, remove_zip=True) -> str:
        """
        Downloads the compressed images set and extracts the contents.

        Arguments:
            remove_zip(bool): remove the compressed file after downloading
                and extracting is complete.

        Returns:
            Path to images directory downloaded.
        """
        image_zip = tf.keras.utils.get_file(fname='train2014.zip',
                                            origin=self.images_origin_url,
                                            extract=True)
        print(f'Downloaded images at: {image_zip}')

        # Clean up the zip file.
        if remove_zip:
            os.remove(image_zip)

        return self.image_dir


class CocoCaptioningDataset:
    """
    Represents the COCO dataset for image captioning.
    """

    def __init__(self,
                 annotation_file: str,
                 image_dir: str,
                 img_load_func: Callable[[str], tf.Tensor] = None,
                 num_words: int = 5000):
        """
        Prepares a COCO captioning dataset.

        Arguments:
            annotation_file: path to annotation file.
            image_dir: path to images directory.
            img_load_func: a function that converts an image path to
                an tf.Tensor instance. Defaults to self.load_img_default
                if None is provided.
            num_words: number of words in the dictionary.
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.num_words = num_words
        self.img_load_func = (self.load_img_default
                              if img_load_func is None else img_load_func)

        print('Loading image and caption data.')
        self.img_ids, self.captions = self._load_data()
        # Sort the data according to image id so that
        # train / val / test sets are not biased.
        self.img_ids, self.captions = zip(
            *sorted(zip(self.img_ids, self.captions)))

        self.num_data = len(self.img_ids)
        # Split the data into train / validation / test sets.
        # The ids returned here represent indices to image id and caption.
        (self.num_train, self.num_val, self.num_test, self.train_ids,
         self.validate_ids, self.test_ids) = self._split_dataset(self.img_ids)

        # Create the tokenizer using the training set.
        print('Fitting tokenizer.')
        train_captions = map(lambda data_id: self.captions[data_id],
                             self.train_ids)
        self.tokenizer = self.create_tokenizer(train_captions)

        print('MsCocoDataset loaded.')

    def __len__(self) -> int:
        return self.num_data

    def _split_dataset(
        self, img_ids: List[int]
    ) -> Tuple[int, int, int, List[int], List[int], List[int]]:
        """
        Split dataset into train, validation and test subsets.

        Arguments:
            img_ids: list of entire image identifiers.

        Returns:
            A tuple of: (num_train, num_val, num_test,
            train_ids, val_ids, test_ids).
        """
        print('Splitting dataset.')
        num_data = len(img_ids)
        data_ids = list(range(num_data))

        # Temporary sizes - we need to determine the actual sizes so that
        # the same image don't belong to more than one split set.
        # This is because there are multiple captions that
        # correspond to the same image.
        tmp_num_test = min(int(num_data * 0.1), 3000)
        tmp_num_val = min(int(num_data * 0.1), 5000)
        tmp_num_train = num_data - tmp_num_test - tmp_num_val

        # Find the position where image ids from train and validation
        # set don't overlap.
        num_train = bisect.bisect(img_ids, img_ids[tmp_num_train - 1])

        train_ids = data_ids[:num_train]
        valtest_ids = data_ids[num_train:]

        # Find the position where image ids from validation and test
        # set don't overlap.
        num_val = bisect.bisect(
            img_ids, img_ids[num_train + tmp_num_val - 1]) - num_train

        # Size of the test set can now be accurately determined.
        num_test = num_data - num_train - num_val

        validate_ids = valtest_ids[:num_val]
        test_ids = valtest_ids[num_val:]

        print(f'* train: {num_train}, validate: {num_val}, test: {num_test}')
        return num_train, num_val, num_test, train_ids, validate_ids, test_ids

    def create_tokenizer(
            self,
            train_captions: List[str]) -> tf.keras.preprocessing.text.Tokenizer:
        """
        Create tokenizer for this dataset.

        Arguments:
            train_captions: list of caption strings

        Returns:
            A tokenizer trained on the provided strings.
        """
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            self.num_words,
            oov_token='<unk>',
            filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(train_captions)
        # Add padding string <pad> to the dictionary.
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        return tokenizer

    @staticmethod
    def load_img_default(img_path: str) -> tf.Tensor:
        """
        Default image load function.
        Reads the image provided by the path and converts to tf.Tensor
        after resizing and preprocessing.

        Arguments:
            img_path: path to image to load

        Returns:
            Image data represented as tf.Tensor.
        """
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img

    def _load_data(self) -> Tuple[List[int], List[str]]:
        """
        image path -> captions mapping
        """
        if not os.path.exists(self.annotation_file):
            raise ValueError(f'{self.annotation_file} does not exist.'
                             'Download the dataset first.')

        if not os.path.exists(self.image_dir):
            raise ValueError(
                f'{self.image_dir} does not exist. Download the dataset first.')

        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        img_ids = []
        captions = []
        for annotation in annotations['annotations']:
            # Add start and end tokens.
            caption = f'<start> {annotation["caption"]} <end>'
            captions.append(caption)

            img_id: int = annotation['image_id']
            img_ids.append(img_id)

        return img_ids, captions

    def img_id_to_path(self, img_id: int) -> str:
        """
        Gives the path to the image given its id.

        Arguments:
            img_id: image identifier

        Returns:
            path: path to the corresponding image
        """
        return os.path.join(self.image_dir, f'COCO_train2014_{img_id:012d}.jpg')

    def _data_ids_to_img_paths(self, data_ids: List[int]) -> List[str]:
        return list(
            map(lambda data_id: self.img_id_to_path(self.img_ids[data_id]),
                data_ids))

    def _data_ids_to_caption_sequences(self, data_ids: List[int]) -> np.ndarray:
        # Convert data ids -> captions
        caption_texts = list(
            map(lambda data_id: self.captions[data_id], data_ids))
        # Convert texts to token index sequences.
        seqs = self.tokenizer.texts_to_sequences(caption_texts)
        seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs,
                                                             padding='post')
        return seqs

    def _to_tf_dataset(self, data_ids: List[int],
                       num_parallel_calls: int) -> tf.data.Dataset:
        img_paths = self._data_ids_to_img_paths(data_ids)
        # seqs is a tokenized caption texts.
        seqs = self._data_ids_to_caption_sequences(data_ids)

        return tf.data.Dataset.from_tensor_slices((img_paths, seqs)).map(
            lambda img_path, seq: (self.load_img_default(img_path),
                                   tf.convert_to_tensor(seq, dtype=tf.int32)),
            num_parallel_calls)

    def as_tf_datasets(
        self,
        num_parallel_calls: int = tf.data.AUTOTUNE
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Gives the COCO captioning dataset as tf.data.Dataset instances.

        Arguments:
            num_parallel_calls: number of parallel calls to make during
                processing. Defaults to tf.data.AUTOTUNE.

        Returns:
            A tuple of (train dataset, val dataset, test dataset).
        """
        return tuple(
            map(  # Convert each ids to tf.data.Dataset.
                lambda data_ids: self._to_tf_dataset(data_ids,
                                                     num_parallel_calls),
                (self.train_ids, self.validate_ids, self.test_ids)))
