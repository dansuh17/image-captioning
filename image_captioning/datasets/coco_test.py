"""
Testing for Coco datasets.
"""
import os
import unittest
from pathlib import Path
import shutil

from .coco import default_cache_dir, CocoAnnotations, CocoImages, CocoCaptioningDataset


def script_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))


class DefaultCacheDirTest(unittest.TestCase):
    """Test default_cache_dir utility function."""

    def test_default_cache_dir(self):
        """Test whether default cache dir matches the expected value."""
        cache_dir = default_cache_dir()
        self.assertEqual(cache_dir,
                         os.path.expanduser('~') + '/.keras/datasets')


class CocoAnnotationsTest(unittest.TestCase):
    """Test functionalities of CocoAnnotations class."""
    test_folder = 'test'

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(script_dir(), cls.test_folder)

    def setUp(self):
        """Create a test-only directory for annotations."""
        Path(self.test_dir + '/annotations').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Remove the test-only directory for annotations."""
        shutil.rmtree(self.test_dir)

    def test_not_exists(self):
        """Test that CocoAnnotations file does not exists."""
        annotations = CocoAnnotations(cache_dir='adsfsf')  # dummy dir
        self.assertFalse(annotations.exists())

    def test_exists(self):
        """Test that CocoAnnotations file exists."""
        annotations = CocoAnnotations(cache_dir=self.test_dir)
        Path(annotations.path()).touch()  # create the target file
        self.assertTrue(annotations.exists())

    def test_path(self):
        """Test the path string matches."""
        annotations = CocoAnnotations(cache_dir='abcdef')  # dummy dir
        self.assertEqual(annotations.path(),
                         'abcdef/annotations/captions_train2014.json')


class CocoImagesTest(unittest.TestCase):
    """Test fnctionalities of CocoImages class."""
    test_dir = 'test'

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(script_dir(), cls.test_dir)

    def setUp(self):
        """Create a test-only directory for images."""
        Path(self.test_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Remove the test-only directory."""
        shutil.rmtree(self.test_dir)

    def test_not_exists(self):
        """Test that CocoImages file does not exists."""
        coco_images = CocoImages(cache_dir='sdfsd')  # dummy dir
        self.assertFalse(coco_images.exists())

    def test_exists(self):
        """Test that CocoImages file exists."""
        coco_images = CocoImages(cache_dir=self.test_dir)
        Path(coco_images.path()).mkdir()  # create the target dir
        self.assertTrue(coco_images.exists())

    def test_path(self):
        """Test the path string matches."""
        coco_images = CocoImages(cache_dir='aaa')  # dummy dir
        self.assertEqual(coco_images.path(), 'aaa/train2014')


class CocoCaptioningDatasetTest(unittest.TestCase):
    """Test functionalities of CocoCaptioningDataset."""
    testfile_dir = 'testfiles/coco'
    # Dummy annotation file only for testing.
    test_annotation_file = 'test_annotations.json'
    # Dummy image files only for testing.
    test_image_folder = 'images'

    @classmethod
    def setUpClass(cls):
        # Get the directory where testfiles are located.
        cls.test_dir = os.path.join(script_dir(), cls.testfile_dir)

    def test_img_id_to_path(self):
        """Test the image id to path conversion."""
        annotation_file = os.path.join(self.test_dir, self.test_annotation_file)
        image_dir = os.path.join(script_dir(), self.testfile_dir,
                                 self.test_image_folder)
        coco_dataset = CocoCaptioningDataset(annotation_file, image_dir)

        img_path = coco_dataset.img_id_to_path(img_id=1)

        self.assertEqual(
            img_path,
            os.path.join(self.test_dir, self.test_image_folder,
                         'COCO_train2014_000000000001.jpg'))

    def test_len(self):
        """Test the dataset size."""
        annotation_file = os.path.join(self.test_dir, self.test_annotation_file)
        image_dir = os.path.join(script_dir(), self.testfile_dir,
                                 self.test_image_folder)
        coco_dataset = CocoCaptioningDataset(annotation_file,
                                             image_dir,
                                             num_words=20)

        self.assertEqual(len(coco_dataset), 12)

    def test_as_tf_datasets(self):
        """Test as_tf_datasets()."""
        annotation_file = os.path.join(self.test_dir, self.test_annotation_file)
        image_dir = os.path.join(script_dir(), self.testfile_dir,
                                 self.test_image_folder)
        coco_dataset = CocoCaptioningDataset(annotation_file,
                                             image_dir,
                                             num_words=20)

        train_ds, val_ds, test_ds = coco_dataset.as_tf_datasets()
        self.assertEqual(len(train_ds), 10)
        self.assertEqual(len(val_ds), 1)
        self.assertEqual(len(test_ds), 1)
