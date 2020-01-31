# Instance Segmentation using Mask R-CNN

### Data

The images have to segmented into these 5 types of WBC's:
1. Basophil 
2. Eosinophil
3. Neutrophil
4. Lymphocyte
5. Monocyte

### Training the model

1. Download/fork Matterport's [Mask R-CNN](https://github.com/matterport/Mask_RCNN).
2. Download the training images and divide them into train and validation set.
3. In the root directory of Mask R-CNN creating a folder named WBC consisting of images and their corresponding masks. It's structure should be as follows:
```bash
    WBC
    ├──train(same for val)
    │   ├──image
    │   │   ├──Basophil
    │   │   │   ├──Basophil_01.png
    │   │   │   └── ...
    │   │   ├──Eosinophil
    │   │   │   ├──Eosinophil_01.png
    │   │   │   └── ...
    │   │   .
    │   │   .
    │   │   .   
    │   ├──mask
    │   │   ├──Basophil
    │   │   │   ├──Basophil_01.png
    │   │   │   └── ...
    │   │   ├──Eosinophil
    │   │   │   ├──Eosinophil_01.png
    │   │   │   └── ...
    │   │   .
    │   │   .
    └── └── .
```
4. Download the pre-trained COCO [weights](https://github.com/matterport/Mask_RCNN/releases)(mask_rcnn_coco.h5) and save them in the root directory of Mask R-CNN.
5. Also save the `WBC.py` file in this repository into the Mask R-CNN folder.
6. To start training, open terminal in the folder and write
`python3 WBC.py train --dataset=WBC --weights=coco`

### Evaluation of the model

The model was trained for 75 epochs with 60 steps per epoch. The following are some predictions of the model on images in the **validation** set:

![Screenshot from 2019-12-22 14-38-11](https://user-images.githubusercontent.com/47391270/71446932-e029f580-274e-11ea-9492-69201170db54.png)


### Requirements:
*numpy
*scipy
*Pillow
*cython
*matplotlib
*scikit-image
*tensorflow>=1.3.0
*keras>=2.0.8
*opencv-python
*h5py
*imgaug
*IPython[all]


### References
1. https://github.com/matterport/Mask_RCNN
2. https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
3. https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
4. https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1
5. https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd
