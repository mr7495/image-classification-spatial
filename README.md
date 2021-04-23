# Wise-SrNet: A Novel Architecture for Enhancing Image Classification by Learning Spatial Resolution of Feature Maps

**Source paper:** [10.13140/RG.2.2.11271.93606/2](https://doi.org/10.13140/RG.2.2.11271.93606/2)

Global Average Pooling layer (GAP) is the most common method for compressing the feature map and decreasing the number of model's parameters. 

Since the advent of deep convolutional models for classification tasks, one important matter is the way of connecting the generated feature map to the final classification layer.
Older models like AlexNet and VGG utilized two sets of fully connected layers at the end of their architecture for making classification from the final feature map.

This architecture and the same architectures based on using all the feature map neurons are only appliable on small models and datasets because as the number of model channels or classes rises, these architectures increase the number of model weights significantly. Having too many weights for just one layer can lead the model to overfitting or underfitting in different criteria.

On the other hand, although GAP layers are optimized and decrease the model's weights, but cause losing spatial resolution of feature maps. As they average between the spatial values, they cause data loss and so less efficiency.

We proposed another architecture which is called Wise-SrNet, for solving this problem without increasing computational costs.
Wise-SrNet comes with several ideas:







