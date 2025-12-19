# 走向通用 AGI

在之前的章节，我们已经初步了解了神经网络，那么我们再回头聊聊 AI。AI 是一个宽泛的概念，它由 ANI（Artificial Narrow Intelligence）
和 AGI（Artificial General Intelligence）构成。其中 ANI 指专用领域的 AI，例如智能音箱、自动驾驶、智能工厂等，而 AGI 指通用型 AI，
能够像人类一样处理事情的 AI。随着时代的发展，ANI 在专用领域有了巨大的进步，同样，这也促使 AGI 在有了长足的发展。

同时，随着计算机的发展，我们可以使用计算机模拟越来越多的神经元，于是我们是否可以想像只要计算机能够模拟足够多的
神经元进行运算，它就能够像人一样思考和处理事情，是否 AGI 就得以实现？答案是否定的。首先，神经网络中模拟的神经元
算法只会应用线性回归、逻辑回归等运算，相比人脑的神经元计算能力而言过于简单；其实，现在的生物科学对人脑并没有一个
完整的认识，对人脑的认识还只是一个比较粗浅的阶段，依据这些知识构建起来的计算机神经网络算法肯定无法模拟人类大脑
的工作。

虽然前面否定了如今 AGI 被实现的可能性，但 AGI 在未来还是有希望被实现的，下面有一些有趣的生物学实验能够佐证这一点。

![5-3 brain port](http://www.ai-start.com/ml2014/images/2b74c1eeff95db47f5ebd8aef1290f09.jpg)

这张图是用舌头学会“看”的一个例子。它的原理是：这实际上是一个名为BrainPort的系统，它现在正在FDA (美国食品和药物管理局)
的临床试验阶段，它能帮助失明人士看见事物。它的原理是，你在前额上带一个灰度摄像头，面朝前，它就能获取你面前事物的低分辨
率的灰度图像。你连一根线到舌头上安装的电极阵列上，那么每个像素都被映射到你舌头的某个位置上，可能电压值高的点对应一个暗
像素电压值低的点。对应于亮像素，即使依靠它现在的功能，使用这种系统就能让你我在几十分钟里就学会用我们的舌头“看”东西。

![5-4 sonar](http://www.ai-start.com/ml2014/images/95c020b2227ca4b9a9bcbd40099d1766.png)

这是第二个例子，关于人体回声定位或者说人体声纳。你有两种方法可以实现：你可以弹响指，或者咂舌头。不过现在有失明人士，确实
在学校里接受这样的培训，并学会解读从环境反弹回来的声波模式—这就是声纳。如果你搜索YouTube之后，就会发现有些视频讲述了一个
令人称奇的孩子，他因为癌症眼球惨遭移除，虽然失去了眼球，但是通过打响指，他可以四处走动而不撞到任何东西，他能滑滑板，他可
以将篮球投入篮框中。注意这是一个没有眼球的孩子。

![5-5 haptic belt](http://www.ai-start.com/ml2014/images/697ae58b1370e81749f9feb333bdf842.png)

第三个例子是触觉皮带，如果你把它戴在腰上，蜂鸣器会响，而且总是朝向北时发出嗡嗡声。它可以使人拥有方向感，用类似于鸟类感知
方向的方式。

上述的科学研究都表明，人类的大脑皮层其实可以根据不同的输入信号进行适应，并完成相应的计算功能。例如当人类听觉相关的大脑皮层
与耳朵的信号切断，转而接入眼睛的视觉信号，这部分听觉相关的大脑皮层也可以适应视觉信号，学会“看”东西。所以，如果有一天我们
找到了人类大脑神经元的算法，从而在计算机中进行模拟，也许真正的 AGI 将会被实现。
# 前向传播的一般实现

通常我们不需要自己实现神经网络的代码，只需要直接使用 Tensorflow 或者 Pytorch 等工具即可。
但在这个章节中我们会详细介绍神经网络的前向传播，并进行简单实现。这不仅能够帮助我们更加深
入的理解 Tensorflow 或 Pytorch 是如何实现神经网络的前向传播，还可以帮助我们在遇到相应问题时
应该如何优化和处理。

## 单层前向传播

我们继续之前的咖啡豆烘赔示例。通过向训练好的神经网络模型输入时间和温度，然后预测咖啡豆的烘焙结果（好或坏）。

![5-1 单层前向传播示例](https://youke1.picui.cn/s1/2025/12/07/693575975a016.png)

示例图中显示，神经网络模型的隐藏层有三个神经元，而在咖啡豆烘焙的示例中，这些神经元的激活函数我们都是使用 sigmoid 作为激活函数。

所以从数学表达上我们进行如下表示：

  * **节点 1:**
    $$a^{[1]}_1 = g(w^{[1]}_1 \cdot \vec{x} + b^{[1]}_1)$$
  * **节点 2:**
    $$a^{[1]}_2 = g(w^{[1]}_2 \cdot \vec{x} + b^{[1]}_2)$$
  * **节点 3:**
    $$a^{[1]}_3 = g(w^{[1]}_3 \cdot \vec{x} + b^{[1]}_3)$$

对于输出层也是同样的道理。

  * **节点 1 (Output):**
    $$a^{[2]}_1 = g(w^{[2]}_1 \cdot \vec{a}^{[1]} + b^{[2]}_1)$$

### 代码片段

按照上面的数学表达，我们可以通过 python 代码对其进行简单的实现。

### 1\. 输入和第一层 (Layer 1)

    ```python
    x = np.array([200, 17])

    # Node 1
    w1_1 = np.array([1, 2])
    b1_1 = np.array([-1])
    z1_1 = np.dot(w1_1, x) + b  # Note: 'b' likely meant to be b1_1
    a1_1 = sigmoid(z1_1)

    # Node 2
    w1_2 = np.array([-3, 4])
    b1_2 = np.array([1])
    z1_2 = np.dot(w1_2, x) + b  # Note: 'b' likely meant to be b1_2
    a1_2 = sigmoid(z1_2)

    # Node 3
    w1_3 = np.array([5, -6])
    b1_3 = np.array([2])
    z1_3 = np.dot(w1_3, x) + b  # Note: 'b' likely meant to be b1_3
    a1_3 = sigmoid(z1_3)

    a1 = np.array([a1_1, a1_2, a1_3])
    ```

### 2\. 第二层（输出层） (Layer 2 - Output)

    ```python
    w2_1 = np.array([-7, 8])  # w^(2)_1 is likely a vector of length 3, if a^(1) is a vector of length 3. The length 2 here might be a conceptual error or simplification in the slide.
    b2_1 = np.array([3])
    z2_1 = np.dot(w2_1, a1) + b2_1
    a2_1 = sigmoid(z1_1)  # Note: This is likely a typo in the slide, it should be sigmoid(z2_1)
    ```

## 前向传播的一般实现

现在我们已经通过 Python 代码实现了一个简单的单层前向传播，但在使用 Tensorflow 的时候，我们知道通常情况下
神经网络模型不会只有一个隐藏层，而且隐藏层也依据要预测的问题不同而有所不同，所以我们需要在单层前向传播实
现的基础上实现一个更加泛化的前向传播算法。

![5-2 multi-layer](https://youke1.picui.cn/s1/2025/12/07/69357e8468e64.png)

根据前面实现的单层前向传播，我们发现 3 个神经元中的向量 w 其实可以写作矩阵 W，标量 b 同样也可以写作矩阵或
向量的形式。

于是从代码角度，我们可以参考 Tensorflow 的用法，对 dense 层进行泛化的 Python 实现，实现代码如下。

```python
def dense(a_in, W, b, g):
    # Determine the number of units (neurons) in this layer
    units = W.shape[1] # Assumes W is shape (input_features, units)
    
    # Initialize the output activation vector
    a_out = np.zeros(units) # [0.0, 0.0, 0.0]
    
    # Iterate through each unit (neuron) in the layer
    for j in range(units): # units: 0, 1, 2
        # Extract the weight vector w for the j-th unit
        w = W[:, j] 
        
        # Calculate the weighted sum (z)
        # z = dot product of input a_in and weight w, plus bias b[j]
        z = np.dot(w, a_in) + b[j]
        
        # Apply the activation function g to get the output activation a_out[j]
        a_out[j] = g(z)
        
    return a_out
```

之前实现的单层前向传播我们就可以调用 `dense` 函数，改写为如下代码：

```python
# Weight matrix W (2x3)
W = np.array([
    [1, -3, 5],
    [2, 4, -6]
]) # 2 by 3

# Bias vector b (1x3)
b = np.array([-1, 1, 2])

# Input activation vector (Layer 0)
a_in = np.array([-2, 4])

# g is implementation of sigmoid
dense(a_in, W, b, g)
```

多隐藏层的前向传播同样也可以通过调用 `dense` 函数实现。

```python
def sequential(x, W1, b1, W2, b2, W3, b3, W4, b4):
    # Layer 1
    a1 = dense(x, W1, b1, g) 
    # Layer 2
    a2 = dense(a1, W2, b2, g) 
    # Layer 3
    a3 = dense(a2, W3, b3, g)
    # Layer 4 (Output)
    a4 = dense(a3, W4, b4, g)
    
    f_x = a4
    return f_x
```

# 神经网络的高效实现

此时，虽然我们已经通过 for 循环实现了一个泛化的隐藏层算法代码，但在神经网络模型的实际应用中这种算法并不够高效。
如今 GPU 等硬件设施为矩阵的计算赋予了更高的效能，如果我们使用矩阵的计算方式优化当前算法，就能轻松提高算法的效率。
所以，让我们回顾一下关于矩阵计算的知识。

## 矩阵乘法

在之前的实现中，我们是通过将向量 a 与向量 w 进行点乘完成计算，如果我们将向量 a 进行转置再与向量 w 进行矩阵乘法
的运算，得到的结果也是等价的，正如下列两个示例。

* **计算式子：** $z = (1 \times 3) + (2 \times 4)$
* **计算结果：** $3 + 8 = 11$

* **表达式：** $z = \vec{a} \cdot \vec{w}$


* **向量 $\vec{a}$：** $\vec{a} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$
* **转置 $\vec{a}^T$：** $\vec{a}^T = [1 \ 2]$
* **表达式：** $z = \vec{a}^T \vec{w}$

同样的道理，这种方式推广至向量 a 与矩阵 W 进行计算也是可行的，正如下列的示例。

* **表达式:** $Z = \vec{a}^T W$
    * $\vec{a}^T$: 向量 $\vec{a}$ 的转置，维度是 **$1 \times 2$**。
    * $W$: 矩阵 $W$，维度是 **$2 \times 2$**。
    * $Z$: 结果矩阵/向量，维度是 **$1 \times 2$**。

矩阵 $W$ 被视为两个列向量 $\vec{w}_1 = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$ 和 $\vec{w}_2 = \begin{pmatrix} 5 \\ 6 \end{pmatrix}$ 的组合。乘积 $Z$ 中的每个元素都是 $\vec{a}^T$ 和 $W$ 中对应列的点积。

* $\text{Z}$ 的第一个元素 ($\vec{a}^T \vec{w}_1$ 的点积)
* **计算式子:** $(1 \times 3) + (2 \times 4)$
* **中间步骤:** $3 + 8$
* **结果:** $11$

* $\text{Z}$ 的第二个元素 ($\vec{a}^T \vec{w}_2$ 的点积)
* **计算式子:** $(1 \times 5) + (2 \times 6)$
* **中间步骤:** $5 + 12$
* **结果:** $17$

* **最终结果表达式:** $Z = [\begin{pmatrix} \vec{a}^T \vec{w}_1 \end{pmatrix} \begin{pmatrix} \vec{a}^T \vec{w}_2 \end{pmatrix}]$
* **最终计算结果:** $Z = [11 \ 17]$


如果我们将上述的矩阵计算规则再进行推广，矩阵 A 与矩阵 W 的计算结果也符合前面实现的神经网络隐藏层算法的计算结果，正如下列示例所示。

* **$A$ 矩阵:** $A = \begin{pmatrix} 1 & -1 \\ 2 & -2 \end{pmatrix}$
* **$A^T$ 矩阵 (行):** $A^T = \begin{pmatrix} 1 & 2 \\ -1 & -2 \end{pmatrix}$ (被分解为两个行向量 $\vec{a}_1^T$ 和 $\vec{a}_2^T$)
* **$W$ 矩阵 (列):** $W = \begin{pmatrix} 3 & 5 \\ 4 & 6 \end{pmatrix}$ (被分解为两个列向量 $\vec{w}_1$ 和 $\vec{w}_2$)
* **乘法表达式:** $Z = A^T W = \begin{pmatrix} \vec{a}_1^T \vec{w}_1 & \vec{a}_1^T \vec{w}_2 \\ \vec{a}_2^T \vec{w}_1 & \vec{a}_2^T \vec{w}_2 \end{pmatrix}$

$Z$ 的四个元素都是 $\text{row} \times \text{column}$ 形式的点积。

| 位置 | 元素代数式 | 详细计算式 | 中间结果 | 最终结果 |
| :---: | :---: | :---: | :---: | :---: |
| **Row 1, Col 1** | $\vec{a}_1^T \vec{w}_1$ | $(1 \times 3) + (2 \times 4)$ | $3 + 8$ | $11$ |
| **Row 1, Col 2** | $\vec{a}_1^T \vec{w}_2$ | $(1 \times 5) + (2 \times 6)$ | $5 + 12$ | $17$ |
| **Row 2, Col 1** | $\vec{a}_2^T \vec{w}_1$ | $(-1 \times 3) + (-2 \times 4)$ | $-3 + (-8)$ | $-11$ |
| **Row 2, Col 2** | $\vec{a}_2^T \vec{w}_2$ | $(-1 \times 5) + (-2 \times 6)$ | $-5 + (-12)$ | $-17$ |

$$Z = \begin{pmatrix} 11 & 17 \\ -11 & -17 \end{pmatrix}$$

## 代码优化

在回顾完矩阵乘法的规则，并且发现矩阵的计算规则能够应用到神经网络的隐藏层算法实现上后，我们还需要了解 Python 代
码如何实现这些计算。

```python
import numpy as np

A = np.array([[1, -1, 0.1],
              [2, -2, 0.2]])

W = np.array([[3, 5, 7, 9],
              [4, 6, 8, 0]])

AT = A.T 

Z = AT @ W

print("Matrix A:\n", A)
print("\nMatrix W:\n", W)
print("\nMatrix A^T (AT):\n", AT)
print("\nResult Z (A^T @ W):\n", Z)
```

在学会通过 Python 代码进行矩阵计算后，我们就可以对前面章节实现的神经网络代码进行优化。

```python
import numpy as np

# Define the input, weights, and bias
AT = np.array([[200, 17]])
W = np.array([[1, -3, 5],
              [-2, 4, -6]])
b = np.array([[-1, 1, 2]])

def dense(AT, W, b, g):
    # Linear combination: z = (AT * W) + b
    # np.matmul is the matrix multiplication operator
    z = np.matmul(AT, W) + b
    
    # Activation: a_out = g(z)
    a_out = g(z)
    
    return a_out
```

# Tensorflow 的训练细节

在自己实现神经网络的隐藏层算法后，我们可以回头再看看通过 Tensorflow 编写的神经网络模型代码，并且将这些代码对
照之前学习的机器学习知识进行理解，这样可以对 Tensorflow 有更加深入和明确的理解。

之前使用 Tensorflow 实现的神经网络模型代码如下：

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])

from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentropy())

model.fit(X, Y, epochs=100)
```

我们可以从三个角度来对照阐述上述代码的意图，以对 Tensorflow 进行更加深入的理解。

首先，从数学角度来阐述

1.  **模型输出 (Model Output / Hypothesis):**
    * $f_{\vec{w},b}(\vec{x}) = ?$

2.  **损失函数 (Loss Function - for one example):**
    * $L(f_{\vec{w},b}(\vec{x}), y)$

    **成本函数 (Cost Function - for $m$ examples):**
    * $$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})$$

3.  ** 训练 (调整 $\boldsymbol{\vec{w}}$ 和 $\boldsymbol{b}$ 来最小化成本函数 $\boldsymbol{J}$)。**

其次，可以从使用 Python 代码实现逻辑回归模型的角度来阐述

```python
# Hypothesis
z = np.dot(w,x) + b
f_x = 1/ (1+np.exp(-z))

# Loss Function
loss = -y * np.log(f_x) \
       - (1-y) * np.log(1-f_x)

# 梯度下降，寻找代价函数最小值
w = w - alpha * dj_dw
b = b - alpha * dj_db
```

最后，我们从使用 Tensorflow 实现神经网络模型的角度来阐述

```python
# Define Hidden Layer and Activation Function
model = Sequential ([
    Dense (...)
    Dense (...)
    Dense (...)
])

# Loss Function
model.compile(loss=BinaryCrossentropy())

# Train Model
model.fit (X, y, epochs=100)
```

在使用 Tensorflow 时，我们需要根据不同的情况指定代价/损失函数，上述代码中使用的是二元交叉熵（Binary Crossentropy），
交叉熵是统计学上的算法，而二元说明这种代价函数是用于二元分类问题。如果你希望编写的神经网络模型是用于一个线性回归
问题，那么需要将代价函数指定为平方差损失函数（MeanSquareError）。最后通过 `model.fit(X, y, epochs=100)` 代码进行
模型训练，也就是在之前我们讲到的寻找代价函数最小值点时的参数值，同时指定 epochs 让模型进行相应次数的训练，以达到
最佳效果。

# 激活函数

在之前的课程中，我们已经学习过两种激活函数，一种是 sigmoid 激活函数，另一种是线性激活函数，它们有各自适用解决的
问题。现在，我们来认识另外一种激活函数**校正线性单元**（Rectified Linear Unit），简称 **ReLU**，函数图如下。

![5-7 ReLU](https://youke2.picui.cn/s1/2025/12/09/69382868da081.png)

在神经网络中，线性激活函数也被称为非激活函数（No Activation Function），因为在线性激活函数中 `g(z) = z`，期望函数
不会被激活函数所改变。与 sigmoid 函数相比，ReLU 的计算速度更快，它不用像 sigmoid 要进行指数等相对复杂的运算，只用
比较 z 与 0 的值即可。从函数曲线上看，sigmoid 函数的平滑区域比 ReLU 的更多，也就代表着进行梯度下降运算时 sigmoid 
会存在更多的局部最小值点，减缓梯度下降的速度。所以从这两个原因出发，在进行机器学习时，ReLU 是一个比 sigmoid 更加
高效的激活函数。

虽然 ReLU 是一个进行机器学习运算时更加高效的函数，但它们有不同的适用场景。在处理二元分类问题时，需要选择 sigmoid 
函数；在处理带有负数线性预测结果的问题时，线性激活函数是更加合适的选择，例如预测股票涨跌时；在处理非负数线性预测
结果时，ReLU 是更加适合的选择，例如预测房价时。所以，对于神经网络的输出层来说，我们需要根据当前解决的问题不同选择
使用合适的激活函数，这样才能达到理想的效果。

在神经网络模型中，除了输出层需要选择激活函数外，在隐藏层也需要选择合适的激活函数，不过隐藏层选择激活函数的逻辑与
输出层不太相同。我们使用一个简化后的神经网络模型作为示例，这个模型只有两个隐藏层，分别都只有一个神经元，每个神经
元的激活函数都是线性激活函数，输入值为一个特征值，于是我们可以得到激活值可以用如下数学式表示。

* **第一层激活 (Layer 1 Activation):**
    $$a^{[1]} = w^{[1]}_{1}x + b^{[1]}_{1}$$

* **第二层激活 (Layer 2 Activation, in terms of $a^{[1]}$):**
    $$a^{[2]} = w^{[2]}_{1}a^{[1]} + b^{[2]}_{1}$$

* **第二层激活 (Layer 2 Activation, in terms of $x$, after substitution):**
    $$a^{[2]} = w^{[2]}_{1} (w^{[1]}_{1}x + b^{[1]}_{1}) + b^{[2]}_{1}$$

* **最终简化形式 (Final Simplified Form - Linear Model):**
    $$\vec{a}^{[2]} = (\vec{w}^{[2]}_{1}\vec{w}^{[1]}_{1}) x + w^{[2]}_{1}b^{[1]}_{1} + b^{[2]}_{1}$$
    * 这里的 $\boldsymbol{\omega}$ 代表复合权重项 $\boldsymbol{(\vec{w}^{[2]}_{1}\vec{w}^{[1]}_{1})}$。
    $$a^{[2]} = \omega x + b$$

从上述示例中我们了解到，当隐藏层的神经元都应用线性激活函数时，神经网络最终被简化为了一个单层的线性模型，
这样会极大限制神经网络模型的表示能力，不利于神经网络模型的应用。如果不使用线性激活函数，基于前面对 sigmoid 函
数与 ReLU 函数的对比，ReLU 函数是作为隐藏层神经元激活函数的首选，而如今的神经网络模型领域也正是如此，多
数情况下人们都选择 ReLU 作为隐藏层的激活函数。当然，如今也有很多其他的函数作为隐藏层的激活函数，但那些是
针对特定问题的选择，是少数情况，如果你感兴趣也可以尝试和研究。

# 多分类

在机器学习需要解决的问题中，除了前面讲到的二元分类问题之外，现实生活中还有很多亟待解决的问题是多分类问题，
此时再继续使用 sigmoid 函数作为输出层的激活函数显然是不合适，所以在多分类问题中需要使用 **softmax 回归**。

![5-8 multiple classes](https://youke2.picui.cn/s1/2025/12/09/6937e41f23f51.png)

## softmax

在二元分类问题中，通常使用下列的数学表达式进行表示。

$$a_1 = g(z) = \frac{1}{1 + e^{-z}} = P(y = 1|\vec{x})$$

$$a_2 = 1 - a_1 = P(y = 0|\vec{x})$$

在多分类问题中，如果需要进行 4 个分类，使用 softmax 回归时数学表达式如下：

$$a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}} = P(y = 1|\vec{x})$$

$$a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}} = P(y = 2|\vec{x})$$

$$a_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}} = P(y = 3|\vec{x})$$

$$a_4 = \frac{e^{z_4}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}} = P(y = 4|\vec{x})$$

我们可以将上述 softmax 的数学表达式进行泛化

$$z_j = \vec{w}_j \cdot \vec{x} + b_j \quad j = 1, \dots, N$$

$$a_j = \frac{e^{z_j}}{\sum_{k=1}^{N} e^{z_k}} = P(y = j|\vec{x})$$

泛化后 softmax 函数其实也适用于二元分类，和之前使用 sigmoid 表示的二元分类问题表达式是一致的，所以我们得到
了一个统一的逻辑回归模型。

* Softmax 概率 ($a$) 的计算:
* $$a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + \dots + e^{z_N}} = P(y = 1|\vec{x})$$
* $$\vdots$$
* $$a_N = \frac{e^{z_N}}{e^{z_1} + e^{z_2} + \dots + e^{z_N}} = P(y = N|\vec{x})$$

* 交叉熵损失 (Crossentropy Loss) 的计算:
* $$\text{loss}(a_1, a_2, \dots, a_N, y) = \begin{cases} -\log a_1 & \text{if } y = 1 \\ -\log a_2 & \text{if } y = 2 \\ \vdots \\ -\log a_N & \text{if } y = N \end{cases}$$

在 softmax 的代价函数中，aj 与代价函数也符合 -log 的函数关系，这会大大刺激算法将 aj 尽可能地增大，而每个训练
样例的计算中，也只是会针对一个 y 的实际取值进行计算，最终也只会计算对应 aj 的负对数值。

## 应用 softmax

在学习了 softmax 后，我们不仅仅像之前一样，只能通过 sigmoid 识别手写数字 0 和 1，而是可以进行更多手写数字的
识别，例如从 1 到 10 的手写数字识别，如下图。

![5-9 softmax multiple classes](https://youke2.picui.cn/s1/2025/12/09/69381a82e911e.png)

* Logits ($z$) 的计算:
* $$z_1^{[3]} = \vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]}$$
* $$\vdots$$
* $$z_{10}^{[3]} = \vec{w}_{10}^{[3]} \cdot \vec{a}^{[2]} + b_{10}^{[3]}$$

* 概率 ($a$) 的计算:
* $$a_1^{[3]} = \frac{e^{z_1^{[3]}}}{e^{z_1^{[3]}} + \dots + e^{z_{10}^{[3]}}} = P(y = 1|\vec{x})$$
* $$\vdots$$
* $$a_{10}^{[3]} = \frac{e^{z_{10}^{[3]}}}{e^{z_1^{[3]}} + \dots + e^{z_{10}^{[3]}}} = P(y = 10|\vec{x})$$

将 softmax 激活函数的数学表达式与 sigmoid 和 ReLU 比较，我们能够发现，softmax 在计算激活值（a）时，需要所有
的 z 值，而不像 sigmoid 和 ReLU 在计算某个激活值时只需要对应的一个 z 值。现在，我们可以使用 Tensorflow 实现
一个以 softmax 作为输出层激活函数的模型。

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='softmax')
])

from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X,Y,epochs=100)
```

其中，代价函数使用的是稀疏类别交叉熵函数（Sparse Categorical Crossentropy）。

## 优化 softmax 模型

在上面的代码中，我们已经成功通过 Tensorflow 编写了一个 softmax 作为输出层激活函数的模型。虽然这个模型确实是
按照正确的写法实现代码，但在实际应用过程中可能出现一些问题。

这个问题的源头来自于计算机对于小数的表达是有精度的，也就是有舍入误差的。现在，我们来简化这个问题，通过 sigmoid 
进行举例。

例如，在应用 sigmoid 前计算得的 z 值的绝对值极大时，应用 sigmoid 计算的结果 y 无限接近于 1，导致二元计算中的反向
结果 `1-y` 会无限接近于 0。此时，由于计算机的舍入误差问题，最终会导致计算 `log(1-y)` 时结果 `1-y` 被当作 0 代入
计算，这样将导致数值下溢，产生负无穷的结果，这使得整个模型的计算结果出现了巨大误差。

除此之外，上述情况还会导致梯度饱和。当 y 无限接近于 1 时，sigmoid 的导数就会趋近 0，也就是说此时的梯度已经极小，
造成梯度饱和，模型将几乎不再更新参数，最终导致模型很难收敛，或无法收敛。

分析完了存在的问题，我们需要知道如何解决，其实非常简单，就是暂时不计算 a 值，而是将 a 值的表达式直接代入激活函数
中进行计算。在使用 Tensorflow 时，只需要在指定代价函数时添加参数 `from_logits=True`，并将输出层的激活函数指定为线
性激活函数，我们自己定义的神经网络模型就只会输出 z 值，最后通过 tensorflow 提供的 softmax 函数应用模型即可。

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear') 
])

from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

model.fit(X,Y,epochs=100)

logits = model(X)
f_x = tf.nn.softmax(logits)
```

# 模型优化

## 学习率优化

对模型进行训练时，主要通过梯度下降的方式找到代价函数的全局小值点，其中学习率决定着梯度下降的速度和结果。当学习率
过小时，虽然最终能够到达代价函数的最小值点，但会需要进行更多步的模型训练，会对时间和硬件资源造成更多消耗。当学习
率过大时，梯度下降会一直振荡，除了模型训练步数增加外，很可能无法到达代价函数的最小值点，影响模型的最终结果。

![alpha optimitizer](https://i.mji.rip/2025/12/17/8532bc41a1e6263ba799f908e43e7ca3.png)

在使用梯度下降时，学习率只能够在初始时由人工给定，要测试出合适的学习率就需要人不断地调试。所以，我们十分需要一个
能够在训练过程中，根据具体情况优化学习率的方法。在机器学习中，前人已经开发了这样的算法，叫做 **Adam (Adaptive Moment estimation)**，
中文名称为**自适应矩估计**算法。

Adam 算法会为不同的参数设置不同的学习率，并在更新参数的过程中观察参数的变化。如果参数一直保持一个方向上的变化，那么
它会认为学习率设置过小，它会适当增加学习率。如果它发现参数一直在振荡，那么它会认为学习率过大，它会适当降低学习率。这
种动态调整学习率的算法，在实际使用中会比梯度下降算法更加高效、灵活。

下列是使用 Adam 算法进行模型训练的代码：

```python
model = Sequential([
    tf.keras.layers.Dense(units=25, activation='sigmoid'),
    tf.keras.layers.Dense(units=15, activation='sigmoid'),
    tf.keras.layers.Dense(units=10, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(X, Y, epochs=100)
```

虽然 Adam 算法会自适应地调整学习率，但我们还在代码中为算法指定了一个初始学习率，设置一个合适的初始学习率会帮助提高模型的
训练速度。

## 隐藏层优化

到目前为止，我们训练的神经网络模型都是使用的稠密层（Dense）构建。通过稠密层也能够构建出规模较大的，能够实际使用的神经
网络模型，不过除了稠密层外神经网络模型还有其他隐藏层选择。

**卷积层（Convolutional Layer）**也是一种神经网络模型的隐藏层选择，它的每个神经元不需要对前一层的激活值全部进行计算，而
只是计算其中的部分值。这种算法除了有效加快模型训练速度的好处外，还能够降低模型过拟合的可能性。下面我们用心电图识别的示例
说明卷积层是如何工作的。

![convolutional layer](https://i.mji.rip/2025/12/17/36356ae16f884fa6403308b7c122eba1.png)

对于卷积层来说，也有一些参数可以调整和配置，比如输入窗口的大小、每一层神经元的个数等。通过这些调配，在一些情况下通过卷积
层构建的神经网络模型可能比使用稠密层构建的神经网络模型更加有效。

