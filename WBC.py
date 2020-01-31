import os
import sys
import numpy as np
import skimage.io
import random
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("/../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class WBCConfig(Config):
    """Configuration for training on the WBC dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "WBC"

    # Running on CPU
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + 5 types of WBC

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 60  
    VALIDATION_STEPS = 2  

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = 'resnet50'

    # Input image resizing
    # In 'square' resizing mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################


class WBCDataset(utils.Dataset):
    
    # function to load the images
    def load_dataset(self, dataset_dir, subset):
        # Adding the 5 classes
        self.add_class("WBC", 1, "Basophil")
        self.add_class("WBC", 2, "Eosinophil")
        self.add_class("WBC", 3, "Lymphocyte")
        self.add_class("WBC", 4, "Monocyte")
        self.add_class("WBC", 5, "Neutrophil")
        
        # Train, validation or test/detect set?
        assert subset in ["train", "val", 'detect']
        self.dataset_dir = os.path.join(dataset_dir, subset)
        
        if subset == 'detect':
            for test in os.listdir(os.path.join(self.dataset_dir, 'image')):
                img_path = os.path.join(self.dataset_dir, 'image', test)
                self.add_image('WBC', image_id=test, path=img_path)
                
        else:
            for filename in os.listdir(os.path.join(self.dataset_dir, 'image')):
                # To get the class ids of the images. 
                if filename == "Basophil":
                    ids = 1
                if filename == "Eosinophil":
                    ids = 2
                if filename == "Lymphocyte":
                    ids = 3
                if filename == "Monocyte":
                    ids = 4
                if filename == "Neutrophil":
                    ids = 5

                # Loop over all the images of 1 type of WBC
                for img in os.listdir(os.path.join(self.dataset_dir, 'image', filename)):
                    img_path = os.path.join(self.dataset_dir, 'image', filename, img)
                    image = cv2.imread(img_path)
                    height, width = image.shape[:2]
                    
                    # Reading the mask of the image to get the number of instances in the image
                    mask_path = os.path.join(self.dataset_dir, 'mask', filename, img)
                    mask = cv2.imread(mask_path)
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
                    ret, labels = cv2.connectedComponents(thresh)
                
                    # labels.max() gives the number of instances in the image
                    class_ids=np.zeros([labels.max()]).astype(int)
                    for i in range(labels.max()):
                        class_ids[i]=ids

                    self.add_image('WBC', image_id=img, path=img_path, width=width, height=height,
                                   class_ids=np.array(class_ids))
    
    
    # function to load the masks            
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(self.dataset_dir, 'mask')

        for filename in os.listdir(mask_dir):
            for mask in os.listdir(os.path.join(mask_dir, filename)):
                mname,png = os.path.splitext(mask)
                iname,jpg = os.path.splitext(info['id'])
                # checking if mask is that of the input image
                if mname == iname:
                    mask_r = cv2.imread(os.path.join(mask_dir,filename, mask), cv2.IMREAD_GRAYSCALE)
                    # dimensions of image and mask are not the same. So we pad the masks with black pixels 
                    mask_r = cv2.copyMakeBorder(mask_r,4,4,4,4,cv2.BORDER_CONSTANT,value=[0,0,0])
                    ret, labels = cv2.connectedComponents(mask_r)
                    mask_merge=[]
                    # Creating a boolean mask for each WBC instance in the mask 
                    for label in range(1,ret):
                        mask = np.array(labels, dtype=np.uint8)
                        mask[labels == label]=True
                        mask[labels != label]=False
                        mask_merge.append(mask)
                     
                    # Stacking the masks along the z-axis
                    fin_mask = np.dstack(mask_merge)

        return fin_mask, info['class_ids']

    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    

############################################################
#  Training
############################################################


def train(model):
    # Training dataset.
    dataset_train = WBCDataset()
    dataset_train.load_dataset('path/to/dataset/','train')
    dataset_train.prepare()
 
    # Validation Dataset
    dataset_val = WBCDataset()
    dataset_val.load_dataset('path/to/dataset/','val')
    dataset_val.prepare()

    print("TRAINING LAYERS...")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=75, layers='3+')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


############################################################
#  Command Line
############################################################


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color mask effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WBCConfig()
    else:
        class InferenceConfig(WBCConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes

        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
