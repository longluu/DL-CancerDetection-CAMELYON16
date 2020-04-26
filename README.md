# Project description
In this project, I try the CAMELYON16 challenge (https://camelyon16.grand-challenge.org/data/) but only use a subset of slides (21/400). It may seem like a super small dataset (even for the original one) but note that each slide is a set of images at different levels of magnifications (up to 9 levels). For each slide, the higher level has double resolution compared to the lower level, meaning the highest resolution is up to 500x or 100,000 x 100,000 pixels. Also, each slide has different number of levels and/or resolution.

# Preprocessing
For this project, I chose **level 3 of each slide (around 10,000 x 10,000)** for the training and testing. Then I reserved 2 slides as test data, which leaves 19 slides for training.

1. Because of a small number of images and the super high resolution of each, an efficient approach is to chop off the slides into several small patches and train a model to classify whether that patch contains tumor cell or not. I use a pretty **small patch (32 x 32)** so that we have more training data and also a good localization of a tumor. The result is around **1.2 million patches** (not a bad dataset size).

2. Another observation is that for many images, some big regions is simply gray background. These area are definitely not informative at all and may add noise into the model training. So instead of using all the regions, I tried to remove the patches that contain only background. At first, I tried intensity thresholding, that is converting patches to gray and remove those with mean intensity too high/too low. However, that did not work well because background patches and patches with cells have high overlapping of mean intensity. Therefore, I use a more efficient method, i.e. **thresholding based on variance across color channel**. For each location on the image, I computed the variance across the color channel and then average across all locations. That turns out to be a pretty good method. I chose a **threshold of 5** (just empirically). That results in around **450,000 patches**.

3. Another note is that the remaining patches after thresholding are highly **imbalanced**, meaning that the number of patches without tumor are almost 10 times the number of patches with tumor. With that concern in mind, I first tried to balanced out the dataset by removing most patches without tumor to match it with the number of tumor patches. The result of that balancing is only around **40,000 patches** left for training. Spoiler: I'll show later that using the imbalanced dataset actually works much better (which makes sense given that we have 10x more data).

4. Final small detail, I split the dataset into training (80%) and validation (20%).

Now, we're ready for some exciting results of training deep (or shallow) networks!

# Custom CNN networks
## Architecture: 
This is are relatively "shallow" neural networks. There are 4 convoluational blocks, each block contains a convolutional layer, batch normalization, drop out, activation and max pooling (2 x 2). I use Adam optimization with default parameters and batch size 128. I set the max number of epoch to 200 and early stopping to 50 epochs.

## Basic model
This is the most basic version. Here is the result.
![Custom_train_v1](/figures/custom_model_relu.png)

Some observations:
* Training loss and accuracy curves converge pretty quickly and are smooth/stable. However the validation loss and accuracy oscillates a lot even towards the end of the training. That signals some instability in the networks despite the seemingly good result (96% accuracy).
* There is a big gap between training and testing curves which clearly indicates the model overfits to data. It's interesting because even a shallow model may overfitting!

## Rescaling input
Then I noted that I didn't normalize the values of input data (intensity values are 0-255). So heeding the advice of many deep learning gurus, I scaled the input to 0-1 range. Here is the result.
![Custom_train_v2](/figures/custom_model_relu_rescaleX.png)

Some observations:
* The model stops earlier.
* The overfitting and oscillation are still there.

## Drop-out
Now I try drop out to help with overfitting. First I set the rate to 0.2.
![Custom_train_v3-1](/figures/custom_model_relu_rescaleX_drop0.2.png)

Then I tried 0.4.
![Custom_train_v3-2](/figures/custom_model_relu_rescaleX_drop0.4.png)

Some observations:
* The model also stops early.
* Drop-out doesn't help. It actually hurts when I increase the rate.

## Data augmentation
Now I try another regul. Flip horizontal and vertical.
![Custom_train_v4-1](/figures/custom_model_relu_rescaleX_dataAugFlip.png)

Add shift left-right, up-down (0.2)
![Custom_train_v4-2](/figures/custom_model_relu_rescaleX_dataAugFlipShift0.2.png)

Add rotation
![Custom_train_v4-3](/figures/custom_model_relu_rescaleX_dataAugFlipShift0.2Rot.png)

Remove rotation, add Shear
![Custom_train_v4-4](/figures/custom_model_relu_rescaleX_dataAugFlipShift0.2Shear.png)

## Test result
![Custom_test_1](/figures/test_accuracy_custom.png)
![](/figures/roc_custom_model.png)
![Custom_test_2](/figures/custom_model.png)

# InceptionV3
## Basic
![Custom_train_v4-1](/figures/InceptionV3_method1.png)
![Custom_train_v4-1](/figures/test_accuracy_Inception_m1.png)
![](/figures/roc_InceptionV3_m1.png)
![Custom_train_v4-1](/figures/InceptionV3_meth1.png)

## Data augmentation
![Custom_train_v4-1](/figures/InceptionV3_method2.png)
![Custom_train_v4-1](/figures/test_accuracy_Inception_m2.png)
![](/figures/roc_InceptionV3_m2.png)
![Custom_train_v4-1](/figures/InceptionV3_meth2.png)

## Imbalanced data (10x more patches without tumor)
![Custom_train_v4-1](/figures/InceptionV3_method3_noDataAug.png)

## Data augmentation + imbalanced data (10x more patches without tumor)
![](/figures/InceptionV3_method3.png)
![](/figures/test_accuracy_Inception_m3.png) ![](/figures/roc_InceptionV3_m3.png)
![](/figures/InceptionV3_meth3.png)

# InceptionResnetV2
## Data augmentation
![](/figures/InceptionResnetV3_method1.png)
![](/figures/test_accuracy_InceptionResnetV2.png) ![](/figures/roc_InceptionResnetV2.png)


# Code
