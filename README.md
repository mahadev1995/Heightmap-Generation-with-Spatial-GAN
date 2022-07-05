
# Heightmap-Generation-with-Spatial-GAN

In this project, a variant of the SGAN [1] is used for generalized automatic generation of height maps. Unlike in DCGAN, the spatial dimension in the SGAN generator is not hardcoded. So, the same pretrained generator can be used to generate images of various spatial dimensions, dropping the need to train the generator again and again.
### Model Schematics
| <img src="https://user-images.githubusercontent.com/51476618/177348385-9fc3484c-7ac5-4f7f-af15-ad1d7c2e5d3c.png" width="400"> | <img src="https://user-images.githubusercontent.com/51476618/177369531-90c49450-0f3c-40ca-92b5-83740be47f01.png" width="400"> | <img src="https://user-images.githubusercontent.com/51476618/177369956-396aac38-da6d-492c-9308-f8aa9c669602.png" width="400"> | 
|:--:| :--:|:--:|
|*Schematic diagram of a spatial GAN.*|*Schematic diagram of a spatial GAN Generator.*|*Schematic diagram of a spatial GAN Discriminator.*|

### Results
|<img src="https://user-images.githubusercontent.com/51476618/177371304-4e7c4b29-6cb5-4c3a-bffc-db996e34276e.png" width="400">|<img src="https://user-images.githubusercontent.com/51476618/177371419-8e93429f-98a2-41a0-8126-1c7ce0da4cd2.png" width="400">|
|:--:| :--:|
|*Sixteen training samples containing the height maps.*|*Generated output of size (64x64) from SGAN generator.*|

|<img src="https://user-images.githubusercontent.com/51476618/177373982-68567fd1-d4b9-48e0-9338-a60a4c4640aa.png" width="828">|
|:--:|
|*Generated output of size (128x128) from SGAN generator.*|

|<img src="https://user-images.githubusercontent.com/51476618/177375309-9f4c660e-d207-4d90-89a2-ae6ed8a448f4.png" width="828"> |
|:--:|
|*Generated output of size (256x256) from SGAN generator.*|

### References
1. N. Jetchev, U. Bergmann, and R. Vollgraf, “Texture synthesis with spatial generative adversarial networks,” 2017.
2. Radford, Alec et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." 2015.

