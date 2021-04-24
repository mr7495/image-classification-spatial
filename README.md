# Wise-SrNet: A Novel Architecture for Enhancing Image Classification by Learning Spatial Resolution of Feature Maps

**Source paper:** [10.13140/RG.2.2.11271.93606/2](https://doi.org/10.13140/RG.2.2.11271.93606/2)

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
