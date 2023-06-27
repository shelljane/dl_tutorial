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


