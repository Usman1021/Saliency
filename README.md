
## [Saliency-based Video Summarization for Face Anti-spoofing]
#### Authors: Usman Muhammad, Mourad Oussalah and Jorma Laaksonen

#### Journal: [Pattern Recognition Letters](https://www.sciencedirect.com/journal/pattern-recognition-letters)
##

### Abstract
The paper presents a video summarization method for face anti-spoofing tasks that aims to enhance the performance and efficiency of deep learning models by leveraging visual saliency. In particular, saliency information is extracted from the differences between the Laplacian and Wiener filter outputs of the source images, enabling identification of the most visually salient regions within each frame. Subsequently, the source images are decomposed into base and detail layers, enhancing representation of important information. The weighting maps are then computed based on the saliency information, indicating the importance of each pixel in the image. By linearly combining the base and detail layers using the weighting maps, the method fuses the source images to create a single representative image that summarizes the entire video. The key contribution of the proposed method lies in demonstrating how visual saliency can be used as a data-centric approach to improve the performance and efficiency of face presentation attack detection models. By focusing on the most salient images or regions within the images, a more representative and diverse training set can be created, potentially leading to more effective models. To validate the method's effectiveness, a simple deep learning architecture (CNN-RNN) was used, and the experimental results showcased state-of-the-art performance on five challenging face anti-spoofing datasets.
