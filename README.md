---
marp: true
theme: default
footer: '2022/2/19'
paginate: true
size: 16:9


---
# Mastering Deep Learning Within a Few Hours

Stanley Zheng, CSE, CUHK


## 
- Introduction
- Problem Formulation
- Implementation
- Experiments


---
# Introduction

Artificial v.s. Intelligence


---
## Intelligence

###
- Human Perception

![bg right 100%](figs/human.png)


---
## Artificial

###
- Machine Learning

![bg right 80%](figs/svm.jpg)


---
## Deep Learning

![](figs/dnn.jpg)


---
### What Can Deep Learning Do?

###
- Classification
- What is this?

![bg right 100%](figs/classification.png)


---
### What Can Deep Learning Do?

###
- Objective Detection
- Where is it?

![bg right 100%](figs/detection.jpg)


---
### What Can Deep Learning Do?

###
- Semantic Segmentation
- Classify every pixel 

![bg right 100%](figs/segmentation.jpg)


---
### What Can Deep Learning Do?

###
- Speech Recognition
- Voice to text 

![bg right 100%](figs/speech.png)


---
### What Can Deep Learning Do?

###
- AIGC: AI-Generated Content
- Text-to-image/image-to-image

![bg right 90%](figs/aigc.png)


---
### What Can Deep Learning Do?

###
- GPT: Generative Pre-trained Transformer 
- Autoregressive model

![bg right 90%](figs/gpt.png)


---
# Problem Formulation

$\overline{\boldsymbol{y}} = \boldsymbol{f}(\boldsymbol{W}, \boldsymbol{x}) = \boldsymbol{f}_M(\boldsymbol{W}_M, \boldsymbol{f}_{M-1}(\boldsymbol{W}_{M-1}, ... ... \boldsymbol{f}_1(\boldsymbol{W_1}, \boldsymbol{x})))$

$\boldsymbol{W} = \mathop{\arg\min}_{\boldsymbol{W}} \mathop{\sum}_{i=1}^{N} L(\overline{\boldsymbol{y}}, \boldsymbol{y})$


---
### Revisiting Classification

###
- Input: $\boldsymbol{x}$ is the image (RGB)
- Output: cat: $\boldsymbol{y} = [1, 0]$; dog: $\boldsymbol{y} = [0, 1]$
- $\overline{\boldsymbol{y}} = \boldsymbol{f}(\boldsymbol{W}, \boldsymbol{x})$

![bg right 100%](figs/classification.png)


---
### Imitating Human Perception

###
- Input: $\boldsymbol{x}$ is the image (RGB)
- 1st hidden layer: $\boldsymbol{x}_1 = \boldsymbol{f}_1(\boldsymbol{x}) = \sigma (\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1)$
- 2nd hidden layer: $\boldsymbol{x}_2 = \boldsymbol{f}_2(\boldsymbol{x}) = \sigma (\boldsymbol{W}_2 \boldsymbol{x}_1 + \boldsymbol{b}_2)$
- ... ...

![bg right 100%](figs/human.png)


---
### Matrix Multiplication

###
- $\boldsymbol{W}_1 \boldsymbol{x}$

![bg right 90%](figs/matmul.png)


---
### Activation Function

###
- $\boldsymbol{f}_1(\boldsymbol{x}) = \sigma (\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1)$

![bg right 100%](figs/activation.png)


---
### Imitating Human Perception

###
- Input: $\boldsymbol{x}$ is the image (RGB)
- 1st hidden layer: $\boldsymbol{x}_1 = \boldsymbol{f}_1(\boldsymbol{x}) = \sigma (\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1)$
- 2nd hidden layer: $\boldsymbol{x}_2 = \boldsymbol{f}_2(\boldsymbol{x}) = \sigma (\boldsymbol{W}_2 \boldsymbol{x}_1 + \boldsymbol{b}_2)$
- ... ...

![bg right 100%](figs/human.png)


---
### Output Function

###
- $\boldsymbol{f}_M(\boldsymbol{x}) = \mbox{softmax} (\boldsymbol{W}_M \boldsymbol{x}_{M-1} + \boldsymbol{b}_M)$
- $softmax (\boldsymbol{x}_M)_i = \frac{x_{M, i}}{\mathop{\sum}_j x_{M, j}}$
- Regarded as probabilities

![bg right 100%](figs/softmax.png)


---
### Multi-layer Perceptrons (MLP)

###
- $\boldsymbol{f}(\boldsymbol{x}) = \boldsymbol{f}_M \circ \boldsymbol{f}_{M-1} \circ ... ... \circ \boldsymbol{f}_1 (\boldsymbol{x})$
- $\boldsymbol{W}_1$, $\boldsymbol{b}_1$, ... ..., $\boldsymbol{W}_M$, $\boldsymbol{b}_M$ are trainable

![bg right 100%](figs/mlp.png)


---
### Loss Function

###
- $L(\overline{\boldsymbol{y}}, \boldsymbol{y})$, indicates the quality 
- Smaller is better $\downarrow$
- Cross entropy is for classification
- e.g. $\boldsymbol{p} = [1, 0]$; $\boldsymbol{q} = [0.8, 0.2]$
- $CE(\boldsymbol{p}, \boldsymbol{q}) = - log(0.8)$

![bg right 80%](figs/ce.png)


---
### Gradient Descent

###
- Iterative algorithm
- $\boldsymbol{W}^{(t)} = \boldsymbol{W}^{(t-1)} - \gamma \frac{\partial L}{\partial \boldsymbol{W}^{(t)}}$
- $\gamma$ is the learning rate
- Chain rule $\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$

![bg right 80%](figs/sgd.png)


---
# Implementation

###
- Convolutional Neural Networks
- Training and Testing
- PyTorch


---
### Convolution

###
- From signal processing
- For feature extraction

![bg right 90%](figs/conv.png)


---
### Convolutional Layers

###
- Convolutional layer
- Preserve 2D structure
- Local perception
- Less weights

![bg right 100%](figs/conv2.png)


---
### Convolutional Neural Networks

###
- Convolutional layer
- Max/Avg. pooling layer 
- Fully connected layer (MLP)

![bg right 100%](figs/conv3.jpg)


---
### Training Neural Networks

###
- Sample a batch of data
- Forward: compute the loss
- backward: update the weights
- $\boldsymbol{W}^{(t)} = \boldsymbol{W}^{(t-1)} - \gamma \frac{\partial L}{\partial \boldsymbol{W}^{(t)}}$

![bg right 100%](figs/training.png)


---
### Testing Neural Networks

###
- Sample a batch of data
- Compute the loss and accuracy

![bg right 100%](figs/training.png)


---
### PyTorch

###
- PyTorch is famous
- Computation graph

    ![](figs/graph.png)

![bg right 100%](figs/pytorch.png)



