# semantic compression (devel)

This is our course project repo for the CS285 class at Berkeley.

## Motivation

There are works and their modified versions, such as [RTAB-Map](https://github.com/introlab/rtabmap), [OctoMap](https://github.com/OctoMap/octomap), [DenseSurfelMapping](https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping) and [Kimera](https://github.com/MIT-SPARK/Kimera), can construct 3-D semantic maps with semantic point clouds. A semantic point cloud consists of 3-D points and their corresponding semantic probability distribution vectors. For instance, a pixel in a semantic point cloud contains its *xyz* spatial position and a *n*-by-1 distribution vector with *n* pre-defined classes in the classification neural network.

An example of such a pixel-wise semantic segmentation neural network is [FCN](https://github.com/wkentaro/pytorch-fcn). It takes an *h*-by-*w*-by-3 RGB image as the input, produces an *h*-by-*w*-by-*n* probability distribution matrix as the intermediate result and finally outputs an *h*-by-*w*-by-*1* semantic image. An illustration can be found [here](https://youtu.be/UdZnhZrM2vQ?t=109).

The final *h*-by-*w*-by-*1* semantic image is obtained by simply taking the class with the highest likelihood in a distribution vector. This approach discards the redundancy in the vector and efficiently compresses the information. However, it cannot deal with situations where the distribution is not representative, e.g. the likelihood value is roughly the same across all classes in the distribution vector. Conversely, a representative vector will have ideally a likelihood of nearly 1 for one class and almost 0 for others.

Thus, we cannot naively take the final semantic image in real-world scenarios because this compression method is vulnerable to noise/misclassifications/uncertainties in the intermediate matrix. A Bayesian fusion process is then needed to diminish the misclassifications in the neural network by combining multiple outputs from the neural network and fuse them. See [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference) for more information. There are chances when the 3-D geometry of the constructed map has to change, e.g. a [loop-closure detection](https://youtu.be/g_wN0Nt0VAU?t=23) is triggered in a SLAM process. It is therefore necessary to keep all old measurements in order to Bayesian fuse and register them onto the new 3-D map.

We would like to, firstly, **find a way of compressing the *h*-by-*w*-by-*n* intermediate probability distribution matrix thus a reduction in the data size**, and secondly, **reform the matrix into a Bayesian fusion-friendly data structure to accelerate the semantic registration thus an increase in the speed of computation**.

## Solutions

1. Embedding to combine multiple classes in a distribution vector by clustering and remapping them into a low-dimensional space.
2. [Superpixel](https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08) to combine multiple pixels of similar distribution vectors to reduce the mount of pixels.
3. Compress across a series of probability distribution matrices. Similar to video compression. See [H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding) and [compression picture types](https://en.wikipedia.org/wiki/Video_compression_picture_types#Bi-directional_predicted_frames/slices_(B-frames/slices)).
4. Extent our reach back to the pixel-wise semantic segmentation neural network itself. Anything inside the neural network is valuable to the compression? In this way, we no longer treat the network as a black box.