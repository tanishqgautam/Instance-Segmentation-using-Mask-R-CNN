# Instance Segmentation using Mask R-CNN

### Training the model

1. Download/fork Matterport's [Mask R-CNN](https://github.com/matterport/Mask_RCNN).
2. Download the training images and divide them into train and validation set.
3. In the root directory of Mask R-CNN creating a folder named WBC consisting of images and their corresponding masks.
4. Download the pre-trained COCO [weights](https://github.com/matterport/Mask_RCNN/releases)(mask_rcnn_coco.h5) and save them in the root directory of Mask R-CNN.
5. Also save the `WBC.py` file in this repository into the Mask R-CNN folder.
6. To start training, open terminal in the folder and write
`python3 WBC.py train --dataset=WBC --weights=coco`

### References
1. https://github.com/matterport/Mask_RCNN
2. https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
3. https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
4. https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1
5. https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd
