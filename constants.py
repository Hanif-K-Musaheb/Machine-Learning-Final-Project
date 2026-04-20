'''constants file'''

import os

#Data facts
NUM_CITIES = 23

#Data Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESIZED_IMAGES_PATH = os.path.join(BASE_DIR, "Resized_Images")
RESIZED_SPLIT_IMAGES_PATH = os.path.join(BASE_DIR, "resized_images_split")


TRAIN_DIR = os.path.join(RESIZED_SPLIT_IMAGES_PATH, "train")
VAL_DIR = os.path.join(RESIZED_SPLIT_IMAGES_PATH, "validation")
TESST_DIR = os.path.join(RESIZED_SPLIT_IMAGES_PATH, "test")

#Training Constants

##Data split
TEST_SPLIT=.10
VALIDATION_SPLIT=.10
TRAIN_SPLIT=.80

##Data Loader
BATCH_SIZE=32

##Trainer
LEARNING_RATE = 0.001
EPOCHS = 5 #how many epochs needed for training
COOLING_TIME=0.2