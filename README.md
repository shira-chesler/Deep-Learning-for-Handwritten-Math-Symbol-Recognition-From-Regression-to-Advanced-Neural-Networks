# Deep-Learning-for-Handwritten-Math-Symbol-Recognition-From-Regression-to-Advanced-Neural-Networks

## Shachar Fridman, Shira Chesler, Yuval Baror

## March 3rd, 2024

```
Abstract
In this essay, we explore machine learning (ML) and deep learning
(DL) techniques for handwritten mathematical symbol recognition.
Using a dataset of over 10,000 images across 19 classes, we assess various
models from softmax regression to convolutional neural networks (CNNs),
enhanced by early stopping and learning rate decay. Our findings highlight
the significant role of data normalization and architectural optimizations
in improving accuracy.
```
## 1 Introduction

Handwritten symbol recognition is pivotal at the intersection of pattern recogni-
tion and artificial intelligence, challenging the realms of optical character recog-
nition, document analysis, and digital assistant systems. The evolution of deep
learning methodologies heralds a new era, enabling the detection and classifica-
tion of handwritten symbols with remarkable accuracy.
The dataset, sourced from Kaggle and encompassing over 10,000 images of hand-
written digits (0-9) and arithmetic operators (including multiplication, addition,
division, subtraction, decimal, equals, and the variables x, y, z), distributed
across 19 unique categories, forms the cornerstone for the assessment of various
multiclass classification models’ efficacy. This exploration extends from softmax
regression, applied to both normalized and unnormalized data, through several
neural network architectures, and culminates in the deployment of a convolu-
tional neural network (CNN). This CNN, further refined with early stopping
and learning rate decay strategies, aims to optimize performance.
To accommodate the diverse resolution of the images, collected at resolutions
from 400x400 to 155x155 pixels, a standardization process resizes all images to a
consistent resolution of 32x32 pixels. Such normalization is essential for uniform
model training and evaluation, ensuring a balanced dataset where each class is
represented by approximately 500 samples. This facilitates a comprehensive


and fair evaluation of deep learning models’ ability to recognize handwritten
mathematical symbols.

## 2 Related work and required background

Recent works in the domain of handwritten mathematical symbol recognition
have showcased remarkable advancements, evidencing a spectrum of methodolo-
gies and their outcomes. Initial forays with Random Forest models yielded an
accuracy of 78% across 16 classes, setting the stage for subsequent innovations.
DenseNet201 (1), with its dense connectivity, significantly improved accuracy to
88%, while the deployment of CNNs pushed performance boundaries even fur-
ther, achieving up to 89% and 99.81% accuracy in specific configurations (using
only 18 classes). Architectural innovations were not limited to CNNs; ResNet
(2) and MobileNet V2 (3) underscored the importance of tailored frameworks,
with the latter achieving a notable 96% accuracy across 19 classes. The journey
didn’t stop there; the fine-tuning of DenseNet alongside CNNs, the strategic
application of transfer learning with EfficientNetB6 (4), and the rigorous train-
ing of CNNs for extended epochs, collectively advanced the state-of-the-art,
reaching accuracies as high as 99.5%.

## 3 Previous Attempts

The initial exploration into machine learning (ML) and deep learning (DL) for
recognizing handwritten mathematical symbols began with softmax regression,
a fundamental classification algorithm. Despite its simplicity compared to more
advanced neural network designs, softmax regression served as a vital baseline
for assessing the effectiveness of subsequent models. With a learning rate of 0.
and trained over 1000 epochs, the softmax regression model achieved a baseline
accuracy of 21.39% on unnormalized data. This initial result highlighted the
need for further adjustments and underscored the importance of preprocessing
steps such as normalization.
The potential for improvement was recognized, and the pixel values of the
input images were normalized by dividing them by 255.0, ensuring that they fell
within the range of [0, 1]. This preprocessing step led to a notable enhancement
in model performance, with the accuracy rising to 40.48%. The improvement
underscored the significance of normalization in improving model convergence
and overall accuracy.
In pursuit of higher classification accuracy, the study transitioned from soft-
max regression to more complex neural network (NN) architectures. Various
NN models were experimented with, focusing on different architectural con-
figurations that varied in the number of hidden units per layer and activation
functions. The general architecture of the NN models involved multiple layers of
neurons interconnected by weighted edges, with each layer applying a nonlinear
transformation to the input data to progressively extract higher-level features.


To optimize the models, the Adam optimizer with a learning rate of 0.
was employed and the models were trained for 1000 epochs. During training,
the loss function’s convergence was monitored and the models’ performance on
both the training and validation sets was evaluated. The table below details
the configurations and performance metrics of these neural network models,
illustrating the progression and insights gained from each architectural variation.

```
Model ||  Description ||  Epochs || Train  Acc || Validation Acc || Test Acc
NN [128, 64] 1000 83.2% 67.8% 67.6%
NN [256, 128] 1000 86.6% 69.7% 70.7%
NN [512, 128, 32] 1000 82.2% 69.4% 69.1%
NN [512, 256, 128] 1000 87.5% 72.5% 71.2%
NN [512, 256, 128, 64] 1000 82.6% 70.8% 69.1%
```
Table 1: Summary of different neural network architectures and their training,
validation, and test accuracies.

Among the neural network architectures explored, the configuration with
[512, 256, 128] hidden units per layer stood out, yielding the best performance.
This model achieved an accuracy of 72.59% on the validation set and 71.23% on
the test set, demonstrating its effectiveness in recognizing handwritten mathe-
matical symbols.

## 4 Project Description

The project adopts a sophisticated approach with a CNN architecture featuring
two convolutional layers, both employing ReLU activation and max pooling, to
adeptly identify the complex patterns of handwritten mathematical symbols.
The initial convolutional layer incorporates 32 filters of 5x5 size, followed by
a second layer with 64 filters, enhancing the model’s capability in feature ex-
traction. A dropout technique is applied post a dense layer of 1024 neurons,
aiming to prevent overfitting. This architecture is fine-tuned using early stop-
ping, learning rate decay, and specific weight initialization, demonstrating a
commitment to accuracy in symbol recognition, all written in TensorFlow 1.

## 5 Experiment Results

The deployment of a convolutional neural network (CNN) represented a signifi-
cant advancement, achieving an initial accuracy of 95.4%. Through the applica-
tion of learning rate decay and early stopping, performance improved to 96.9%,
while substantially reducing the training duration, evidenced by a 25% decrease
in epochs required for peak accuracy. The introduction of specific initialization
strategies further elevated the accuracy to 97.8%, with the model reaching its
optimal state at epoch 489, showcasing the efficiency of these enhancements in
the training process.


The table and subsequent confusion matrix below detail the performance
metrics and classification effectiveness of the CNN model, emphasizing its pro-
ficiency in accurately recognizing various handwritten mathematical symbols.

```
Model || Description || Epochs  || Train Acc || Validation Acc || Test Acc
CNN (2 Conv Layers + Dropout) 2000 99.3% 95.2% 95.4%
CNN + Early Stopping + LR Decay 500 99.6% 96.9% 97.1%
CNN + ES + LR Decay + Init 500 99.7% 97.5% 97.8%
```
Table 2: Summary of model configurations and their training, validation, and
test accuracies.

![test results confusion matrix](https://github.com/shira-chesler/Deep-Learning-for-Handwritten-Math-Symbol-Recognition-From-Regression-to-Advanced-Neural-Networks/blob/main/confusion%20matrix.png)
Figure 1: Confusion matrix for the best-performing CNN model on the test set.
The labels are encoded as follows: 0-9 correspond to digits 0 to 9, 10: ’add’, 11:
’dec’, 12: ’div’, 13: ’eq’, 14: ’mul’, 15: ’sub’, 16: ’x’, 17: ’y’, 18: ’z’.

The confusion matrix validates the model’s effectiveness, offering insights
into its class-by-class performance and highlighting potential areas for further
refinement.


## 6 Conclusions

Systematic experimentation with softmax regression and CNN illustrates the
profound influence of data preprocessing, complex network architectures, and
precise optimization strategies on model efficacy. The use of early stopping and
learning rate decay markedly improved our CNN models’ efficiency and accu-
racy, facilitating the attainment of leading-edge accuracy with fewer training
epochs. This endeavor, primarily educational in nature, emphasizes ML and
DL’s immense potential in pattern recognition, laying a robust groundwork for
future inquiries. It accentuates the critical role of ongoing experimentation and
refinement in enhancing neural network performance for intricate recognition
tasks.

## References

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Densely Connected Convolutional Networks,https://arxiv.org/pdf/1608.
06993.pdf

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian SunDeep Residual Learn-
ing for Image Recognition,https://arxiv.org/pdf/1512.03385.pdf

[3] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-
Chieh ChenMobileNetV2: Inverted Residuals and Linear Bottlenecks,https:
//arxiv.org/pdf/1801.04381.pdf

[4] Mingxing Tan, Quoc V. LeEfficientNet: Rethinking Model Scaling for Con-
volutional Neural Networks,https://arxiv.org/pdf/1905.11946.pdf


