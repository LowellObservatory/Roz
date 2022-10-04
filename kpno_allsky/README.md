This version was cloned 04-October-2022 from 
https://github.com/dylanagreen/kpno-allsky

This directory was subsequently modified by tbowers to meet the needs
of the LDT All-Sky processing

# kpno-allsky

These are some scripts/software based on all-sky images from the Kitt Peak National Observatory. These images can be found at the following url: http://kpasca-archives.tuc.noao.edu

Current dependencies:
- AstroPy
- SciPy
- NumPy
- matplotlib
- Requests
- pyephem

Spacewatch.py requires the additional dependency of pytesseract.


Details on script operation can be found in the [extensive documentation](https://kpno-allsky.readthedocs.io/en/latest/) I've put together.


### Notebooks
The Notebooks folder contains various Jupyter Notebooks that were and are used for plotting, models, and rapid prototyping. In general I've outlined their purpose in the header of the notebook.

# Convolutional Neural Networks
Recently my effort has been focused on designing a CNN that can recognize and pick out the clouds in the all-sky images automatically. The following lists the versioning scheme I've used for these notbooks, entitled "Image Classifier" in the main repository. 

- **v1** - A variation of ResNet designed to classify the entire image as "cloudy" or "not cloudy"
- **v2** - Used a smaller 32x32 first layer network based on [Deep learning for cloud detection](https://hal.archives-ouvertes.fr/hal-01783857/document) to try and map out every cloud pixel in the image.
- **v3** - Uses a Fully Convolutional Version of **v2** to implement the paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) to try and generate a heatmap of cloud pixels.
- **v4** - A short lived version of **v3** that used 64x64 training patches instead of 32x32 to try and get a better sense of ghost images.
- **v5** - The most recent version, which adds two additional layers that skip most of the pooling layers in **v3** to keep a global sense of the training patches. This is a simpler implementation of the skip connections mentioned in the FCN paper above. Additionally changed the network from a two class classifier (No Clouds, Clouds) to a three class classifier (No Clouds, Ghosts, Clouds).