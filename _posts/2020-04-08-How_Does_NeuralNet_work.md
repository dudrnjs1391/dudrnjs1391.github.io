---
layout: post
categories: DeepLearning
title: "Content - 1. How does Neural Net work?"
tags: [Deep Learning, Deep Neural Network]
---


## - How does Neural Network? - Feed Forward

### Prologue에서는 Neural Network가 어디에 사용되고, 어떻게 구성되어 있는지에 대해서 알아보았다. 그렇다면, Neural Network가 어떻게 0과 1을 분류해내고, 
음성을 인식하는 것인지 그 과정을 파헤쳐보자.

- **0~9까지의 숫자 사진이 들어있는 MNIST라는 데이터를 통해 Neural Network가 
사진을 받아들여, 어떤 숫자인지 인식하는 과정을 차근차근 살펴보도록 하자.**
    - **MNIST 데이터셋**은 Neural Network를 입문할 때, 가장 많이 사용되는 데이터로
    Training을 위한 60000개의 사진들과, Test를 위한 10000개의 사진들로 구성된다.
        - MNIST 데이터 내의 각 숫자 그림은 **28*28** 총 **784**개의 Pixel로 이루어져 있다.
        - 숫자 그림 내의 각각의 Pixel들은 **0~255**값의 밝기로 이루어져 있다.
        - 이 Pixel들이 0~9중 어떠한 수인지 알려주는 **정답**이 존재한다.

    ![Content%201%20How%20does%20Neural%20Net%20work/prologue-3-0.png](Content%201%20How%20does%20Neural%20Net%20work/prologue-3-0.png)

    - 이 Pixel들을 255로 나누면 **0~1 사이의 값 (***0/255 = 0*)을 가지게 되면서, 
    **1에 가까울수록 밝아지며, 0에 가까울수록 어두워진다**.
        - **왜** 255로 나누어야 하는지에 대해서는 *실습 챕터* 에서 설명하겠다.

---

- **Neural Network와 사람의 뇌는 정보를 조각내고 조합하는 과정을 통해 물체를 인식한다.**
    - 뇌는 물체를 인식할 때 여러 정보들을 **분할**하고 **전기신호를 통해 조합하여 인식**한다.
        - Neural Network도 이와 같이 여러 정보들을
        ①. 정보들을 분할하여 ***Input Layer***에서 받아들이고,
        ②. ***Hidden Layer***에서 이를 조합하여,
        ③. ***Output Layer***에서 인식하는 과정을 거친다.
        - 예를 들어, 6을 인식할 때는 **1자형**의 긴 Pixel들과 **O자형**의 동그라미 Pixel로 분할해
        이를 결합하고 최종적으로 6이라는 숫자로 인식할 것이다.

    - 하지만 1자형의 긴 Pixel로 인해 1이나 4로 예측할 수도, O자형의 동그라미 Pixel로 인해 
    8이나 0으로 예측할 수도 있을텐데 어떻게 6으로 인식하는 것일까?

    ![Content%201%20How%20does%20Neural%20Net%20work/prologue-3-2.gif](Content%201%20How%20does%20Neural%20Net%20work/prologue-3-2.gif)

    ---

1. **본격적으로, Neural Network에 Pixel을 넣어보자.** 
    - 먼저, 위 그림은 784개의 Pixel로 조각나 있다. 이 Pixel들은 ***Input Layer***로 들어가게 된다.

    ---

2. **들어간 Pixel들은 Hidden Layer에서의 어떻게 처리 될까?**
    - 아래와 같이 노란 네모칸을 ***Hidden Layer*** 안에 있는 하나의 ***node***라고 가정해보자.

    ![Content%201%20How%20does%20Neural%20Net%20work/Content-1-2.png](Content%201%20How%20does%20Neural%20Net%20work/Content-1-2.png)

    - 784개의 Pixel들은 위치와 밝기, 모양 등에 따라 Hidden Lyaer 안의 node로 압축되면서, 
    **가중치("*weight*")**가 곱해지게 된다.
        - 위 Pixel 그림에서
        ①. 노란색 부분에는 **높은 값의 *weight***를**,** 
        ②. 검은색 부분에는 **낮은 값의 *weight***를 곱해주면 
        ③. **1자형**의 긴 Pixel들을 의미하는 node는 최대값이 되며, 
        ④. **6이라는 숫자를 인식**하는데 도움을 줄 수 있다.
        - 가중치에 대한 예시를 더 들어보면, 우리가 어떤 사람을 알아볼 때 *코가 비뚤어진 사람*은 
        눈, 귀, 입보다 **코**를 더 보게 되고, *눈이 큰 사람*은 **눈**을 더 보게되는 것과 같다.

            ![Content%201%20How%20does%20Neural%20Net%20work/Content-1-3.png](Content%201%20How%20does%20Neural%20Net%20work/Content-1-3.png)

            각 부위에 weight를 주고 **얼굴**에 대한 node 값이 최대가 되게 한다.

        - 즉, 위의 하나의 node가 뜻하는 것이 얼굴이라고 가정할 때, 각각의 부위에 ***weight***가 
        곱해지며 얼굴이라는 node의 값이 **최대**가 될 때, 그 사람을 인식할 수 있게 된다.
        - 위 과정은 다음과 같은 식으로 표현할 수 있다.

            $$w_1x_1+w_2x_2+w_3x_3+w_4x_4+...+w_nx_n$$

    - *Hidden Layer* 안에 있는 많은 *node*들은 **0~9까지의 숫자를 맞추는 데 도움을 주는데,
    이러한 *node*들이 **활성화**되게 하기 위해서 "***bias***"라는 값을 더해준다.
        - 6이라는 숫자를 맞출 때, 6을 설명하는 *node*의 **계산식이 **얼마나 엄격해야 하는지
        임계치의 역할을 한다.
        - 이는 다음과 같은 식으로 정리할 수 있다.

            $$w_1x_1+w_2x_2+w_3x_3+w_4x_4+...+w_nx_n+b$$

     

    ---

3. **처리된 node들이 Out layer에서 어떻게 0~9가 되는지 알아보자.**
    - 이렇게 계산된 *Hidden layer* 안의 *node*의 값들은 -부터 +까지 다양한 값을 가지게 된다. 
    하지만, 우리가 원하는 것은 **0~9** 중 어떠한 수인지에 대해 알고자 하는 것이다.

    - -부터 +까지의 *node*의 값들을 우리가 원하는 숫자로 변경하기 위해 
    활성화 함수(***activation function***)라는 것을 사용할 수 있다.
        - 활성화 함수는 쓰임새에 따라 다양한 종류가 있는데,
        - **여러 개의 값 중 어떠한 값**인지에 대한 확률로 바꿔주는 "***softmax***"라는 함수가 있고,

            $$softmax(x)=\frac{e^{x_i}}{\Sigma^K_{k=1}e^{x_k}} \;\; for \;i=1,...,K$$

        - **Yes or No 등의 두가지 확률**(**0~1**)로 바꿔주는 "***sigmoid***"라는 함수 등이 있다.

            ![Content%201%20How%20does%20Neural%20Net%20work/prologue-3-4.png](Content%201%20How%20does%20Neural%20Net%20work/prologue-3-4.png)

    - ***softmax***라는 활성화 함수를 사용하면 ***weight***와 ***bias***를 받은 Pixel들이 0~9까지 중에서 
    어떤 숫자를 나타내는지에 대한 **확률**을 구할 수 있다.
        - 지금까지의 과정을 다음과 같은 수식으로 표현할 수 있다.

        $$\sigma(w_1x_1+w_2x_2+w_3x_3+w_4x_4+...+w_nx_n+b)\\ \sigma=activation\;function$$

    ---

- **지금까지 나온 이야기들이 어렵다고 느껴질 것이다. 하지만, 어렵지 않다!
아래의 예시를 통해 직접 계산해보면서 위의 과정을 차근차근 다시 정리해보자.**
    - ①. 각 Pixel들이 0~9의 숫자 중 어떤 수인지를 알고자 하기 위해서 아래와 같은 Network를 
         구성한다고 하자. 각 Hidden layer에는 ***sigmoid*** ***function***을, output layer에는 ***softmax 
          function***을 사용하며 임의의 weight, bias를 통해 아래 과정을 진행한다.

        ![Content%201%20How%20does%20Neural%20Net%20work/Content-1-6.png](Content%201%20How%20does%20Neural%20Net%20work/Content-1-6.png)

    - ②. 먼저 들어온 Pixel들에 weight를 각각 곱해준 후, bias를 더하여 ***sigmoid*** 함수에 넣으면
          Hidden layer의 두 node는 다음과 같이 계산된다.

        $$node1 = \frac{1}{1+exp^{-((0.8*1+0.6*0.88+ 0.3* 0.47+0.1*0.03)+0.7)}}=0.6736\\ node2 = \frac{1}{1+exp^{-((0.3*1+0.3*0.88+ 0.5* 0.47+0.7*0.03)+0.5)}}=0.6308\\$$

    - ③. 이렇게 계산된 node의 값을 Output layer로 보내면 또 다시 weight와 bias가 계산되면서,
          ***softmax*** 함수를 적용하게 되고 각 숫자에 맞는 **확률값**이 산출된다.

    $$node\ for\ 6 = \frac{exp^{((0.2*0.6738+0.6*0.6308)+2.0)}}{\Sigma^n_{i=0}e^{(wx+b)}}=0.75 \\ node\ for\ 7 =\frac{exp^{((0.4*0.6738-0.1*0.6308)+1.0)}}{\Sigma^n_{i=0}e^{(wx+b)}}=0.20\\node\ for\ 8=\frac{exp^{((-0.2*0.6738+0.3*0.6308)-0.3)}}{\Sigma^n_{i=0}e^{(wx+b)}}=0.05$$

    - 이러한 과정들이 아래 이미지와 같이 계산되는 것을 볼 수 있다.

        ![Content%201%20How%20does%20Neural%20Net%20work/Content-1-6.gif](Content%201%20How%20does%20Neural%20Net%20work/Content-1-6.gif)

    - ④. **자! 위 예시와는 다르게** 아래와 같이 weight와 bias를 변경해 계산해보도록 하자.
         이러한 경우, 6을 잘 맞추지 못하는 것을 확인할 수 있다.

        ![Content%201%20How%20does%20Neural%20Net%20work/Content-1-8.png](Content%201%20How%20does%20Neural%20Net%20work/Content-1-8.png)

    - ⑤. 위와 같은 과정을 통해, 입력된 Pixel들이 여러 weight와 bias를 거치면서  각각 계산되어 
          Pixel들이 0~9 중 어떠한 수인지 인식할 수 있게 된다.

        ![Content%201%20How%20does%20Neural%20Net%20work/prologue-3-7.gif](Content%201%20How%20does%20Neural%20Net%20work/prologue-3-7.gif)

---

- ***In Conclusion :***
지금까지 우리가 알아본 Input Layer와 Hidden Layer를 거쳐 Output Layer까지의 **과정**을 
***Feed-Forward*** 라고 하며, 들어온 Pixel들이 **여러 Hidden Layer 내의 *weight*와 *bias*에** 따라
**어떤 숫자인지를 예측하는 데 영향을 받는다는** 것을 알게 되었다. 

**결국,** 지금까지 진행한 과정은 결국 ***weight*와 *bias*가 어떻게** 적절히 ****구성되는지에 따라 
**숫자를 잘 인식한다는 것을 알게 하고자 하는 과정**이라고 할 수 있다.
그렇다면, *weight*와 *bias*를 어떻게 적절하게 구성할 수 있는지 **Content -2** 에서 알아보도록 하자.

    import python

- 다음 강의:

[Content - 2. How can Neural Network be trained?](https://www.notion.so/Content-2-How-can-Neural-Network-be-trained-31f50851a79c47bd8941acd1f402a0e8)