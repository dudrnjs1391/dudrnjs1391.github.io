---
layout: post
categories: DeepLearning
title: "Prologue - 2. What is Neural Network?"
tags: [Deep Learning, Deep Neural Network]
---


## - So then, What is Neural Network?

- 우리가 튤립을 볼 때 시각, 청각, 후각, 촉각 등을 통해 정보를 받아들이게 된다.
- 이러한 정보들은 뇌 내의 있는 ***Neuron***을 통해 뇌에 입력된다.
- ***Neuron***을 통해 받아들여진 여러 정보들은 일련의 처리를 거쳐 
우리가 본 꽃이 클로버도 국화도 아닌 튤립이라고 인식하게 된다.

    ![Prologue%202%20What%20is%20Neural%20Network/prologue-2-0.png](Prologue%202%20What%20is%20Neural%20Network/prologue-2-0.png)

    Neural Network Image [processed from pngtree.com](https://pngtree.com/so/%EC%B2%98%EB%A6%AC)

- 뇌에서의 일련의 처리 과정을 시스템적으로 본따 구성한 것이 바로 ***Neural Network***이다.

---

### - What Can we do with Neural Net?

- ***Image Classification***

    ![Prologue%202%20What%20is%20Neural%20Network/prologue-2-1.gif](Prologue%202%20What%20is%20Neural%20Network/prologue-2-1.gif)

    - **0과 1의 사진 분류해보기**

- ***Speech Recognition***

![Prologue%202%20What%20is%20Neural%20Network/prologue-2-2.gif](Prologue%202%20What%20is%20Neural%20Network/prologue-2-2.gif)

- **"Soda"이라는 단어 인식해보기**
- 구글 ***Teachable Machine***에서 직접 해보았다. [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/)
- 위에서 직접 해본 이미지 인식은 매우 정확하게 0과 1을 구분해내며,
 음성 인식은 "**Soda**"이라는 단어를 100%로 잡아내는 것을 확인할 수 있다.
- 이러한 원리로 아래처럼 차량 파손 인식, 음성 인식 AI 등 다양한 분야에 
***Neural Network***를 적용할 수 있다.

    ![Prologue%202%20What%20is%20Neural%20Network/prologue-2-3.png](Prologue%202%20What%20is%20Neural%20Network/prologue-2-3.png)

---

### - Structure of Neural Network

- 먼저, Neural Network는 **어떻게 구성**되어 있는지 간략히 살펴보자.
    - Neural Network는 여러 개의 층("***layer***")으로 이루어져 있으며, 
    각 층들은 정보를 보관하고 있는 "***node***"라는 원으로 구성되어 있다.
    - **이미지, 음성, 텍스트** 등의 정보를 Neural Network에서 받아들일 때, 이들이 처음 들어가게 되는 
    Neural Network의 층을 "***Input Layer"***라고 하며, 들어온 정보들이 튤립인지, 클로버인지, 
    또는 0,1 중 어떤 숫자인지 등의 **결과가 결정**되는 마지막 *layer*를 "***Output layer***"라고 한다.
    - 중간에 있는 여러 층들을 "***Hidden Layer***"라고 하며 입력된 정보들이 어떠한 결과를 
    나타내도록 **학습하는 데 도움**을 주게 된다.

        ![Prologue%202%20What%20is%20Neural%20Network/prologue-2-2.png](Prologue%202%20What%20is%20Neural%20Network/prologue-2-2.png)

- ※ 여러 개의 ***Hidden Layer***를 쌓아 ***Deep***하게 만든 Neural Network를 "***Deep Neural Network***"라고 한다. (a.k.a ***Deep Learning***)

---

- 다음 강의: