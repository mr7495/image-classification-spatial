# Wise-SrNet: A Novel Architecture for Enhancing Image Classification by Learning Spatial Resolution of Feature Maps

**Source paper:** [arxiv:2104.12294](https://arxiv.org/abs/2104.12294)

The paper aims to fix the spatial resolution loss problem caused by Global Average Pooling layers. The final introduced architecture is called Wise-SrNet, which enables the model to create the classification array from the feature map without losing data and also keeping almost the same computational cost.

Three image classification benchmarks were studied in this paper:

1-A selected portion of the ImageNet dataset, including 70 classes </br> (https://www.kaggle.com/mohammadrahimzadeh/imagenet-70classes)

2-Intel Image Classification Challenge  </br> (https://www.kaggle.com/puneet6060/intel-image-classification) </br> 

3-MIT Indoors Scenes  </br> (https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019) </br> 

**Part of the code for training Xception with Global Average Pooling layer on 512x512 images:**

```
shape=(512,512,3)
input_tensor=keras.Input(shape=shape)
base_model=keras.applications.Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)
gavg=keras.layers.GlobalAveragePooling2D()(base_model.output)
preds=keras.layers.Dense(67,activation='softmax',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.01),
                          bias_initializer=keras.initializers.Zeros(),)(gavg)
model=keras.Model(inputs=base_model.input, outputs=preds) 
```

**Part of the code for training Xception with Wise-SrNet on 512x512 images:**

```
shape=(512,512,3)
input_tensor=keras.Input(shape=shape)
base_model=keras.applications.Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)
avg=keras.layers.AveragePooling2D(3,padding='valid')(base_model.output)
depthw=keras.layers.DepthwiseConv2D(5,
                                      depthwise_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.01),
                                      bias_initializer=keras.initializers.Zeros(),depthwise_constraint=keras.constraints.NonNeg())(avg)
flat=keras.layers.Flatten()(depthw)
preds=keras.layers.Dense(67,activation='softmax',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.01),
                          bias_initializer=keras.initializers.Zeros(),)(flat)
model=keras.Model(inputs=base_model.input, outputs=preds)  
```

**In all the attached codes for training with various architectures, if you wish to use a different model like NasNet instead of Xception, you must replace the Xception with NASNetLarge in the next line of the code:**

```
base_model=keras.applications.Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)

base_model=keras.applications.NASNetLarge(input_tensor=input_tensor,weights='imagenet',include_top=False)
```

# Classification codes:

In this part, the classification codes for running on the mentioned datasets have been shared.</br> 
Each architecture is fully explained in the paper.

**Classification codes on the Selected portion of the ImageNet dataset using ResNet50 and 224x224 images:** 

ResNet50+GAP: [Sub_ImageNet_ResNet50_GAP_224.ipynb](Sub_ImageNet_ResNet50_GAP_224.ipynb) </br> 
ResNet50+GAP+DP: [Sub_ImageNet_ResNet50_GAP_dp(0_5)_224.ipynb](Sub_ImageNet_ResNet50_GAP_dp(0_5)_224.ipynb)</br> 
ResNet50+Depthw: [Sub_ImageNet_ResNet50_Depthw_224.ipynb](Sub_ImageNet_ResNet50_Depthw_224.ipynb)</br> 
ResNet50+Depthw+constraint: [Sub_ImageNet_ResNet50_Depthw_constraints_224.ipynb](Sub_ImageNet_ResNet50_Depthw_constraints_224.ipynb)</br> 
ResNet50+pre-avg+Depthw+constraint(Wise-SrNet): [Sub_ImageNet_ResNet50_avg_Depthw_constraints_224.ipynb](Sub_ImageNet_ResNet50_avg_Depthw_constraints_224.ipynb)</br> 
ResNet50+pre-avg+Depthw+constraint+DP(Wise-SrNet with dropout): [Sub_ImageNet_ResNet50_avg_Depthw_constraints_dp(0_5)_224.ipynb](Sub_ImageNet_ResNet50_avg_Depthw_constraints_dp(0_5)_224.ipynb)


**Classification codes on the Intel image classification dataset using DenseNet169 and 224x224 images:** 

DenseNet169+GAP: [Intel_DenseNet169_GAP_224.ipynb](Intel_DenseNet169_GAP_224.ipynb)</br> 
DenseNet169+Depthw+constraint: [Intel_DenseNet169_depthw_constaints_224.ipynb](Intel_DenseNet169_depthw_constaints_224.ipynb)</br> 
DenseNet169+pre-avg+Depthw+constraint(Wise-SrNet): [Intel_DenseNet169_avg_depthw_constaints_224.ipynb](Intel_DenseNet169_avg_depthw_constaints_224.ipynb)</br> 

**Classification codes on the MIT Indoors Scenes dataset using Xception and 224x224 images:** 

Xception+GAP: [MIT_Xception_GAP_224.ipynb](MIT_Xception_GAP_224.ipynb)</br> 
Xception+GAP+DP: [MIT_Xception_GAP_dp(0_5)_224.ipynb](MIT_Xception_GAP_dp(0_5)_224.ipynb)</br> 
Xception+Depthw+constraint: [MIT_Xception_depthw_constraints_224.ipynb](MIT_Xception_depthw_constraints_224.ipynb)</br> 
Xception+pre-avg+Depthw+constraint(Wise-SrNet): [MIT_Xception_avg_depthw_constraints_224.ipynb](MIT_Xception_avg_depthw_constraints_224.ipynb)</br> 
Xception+pre-avg+Depthw+constraint+DP(Wise-SrNet with dropout): [MIT_Xception_avg_depthw_constraints_dp(0_5)_224.ipynb](MIT_Xception_avg_depthw_constraints_dp(0_5)_224.ipynb)

**Classification codes on the MIT Indoors Scenes dataset using Xception and 512x512 images:** 

Xception+GAP: [MIT_Xception_GAP_512.ipynb](MIT_Xception_GAP_512.ipynb)</br> 
Xception+Flatten+FC: [MIT_Xception_flatten_FC_512.ipynb](MIT_Xception_flatten_FC_512.ipynb)</br> 
Xception+pre-avg+Depthw+constraint(Wise-SrNet): [MIT_Xception_avg_depthw_constraints_512.ipynb](MIT_Xception_avg_depthw_constraints_512.ipynb)</br> 


Our experiments revealed a very good improvement on 224x224 images and a significant improvement on 512x512 images. Whatever the images are larger, and the number of classes is more, our architecture shows more increment in accuracy than the Global Average Pooling. For more details about the usage of transfer learning, please read the paper.


**For using our proposed methods, please cite it by:**
 ```
@article{rahimzadeh2021wise,
  title={Wise-SrNet: A Novel Architecture for Enhancing Image Classification by Learning Spatial Resolution of Feature Maps},
  author={Rahimzadeh, Mohammad and Parvin, Soroush and Safi, Elnaz and Mohammadi, Mohammad Reza},
  journal={arXiv preprint arXiv:2104.12294},
  year={2021}
}
 ```

 If you have any questions, contact me by this email : mr7495@yahoo.com
