# Neural net basics

## Multi-layer perceptrons

- **multi-layer perceptron**: input layer, 하나 이상의 hidden layer, output layer로 이뤄진 fully-connected network다
    - 흔히 "feed-forward network"와 같은 뜻으로 쓴다. 엄밀히 FFN은 정보가 한 방향으로 흐르는 network 전체를 가리키는 더 넓은 범주다
- neuron 하나는 입력의 weighted sum을 구하고 bias를 더한 뒤 activation function에 통과시킨다
    - $\mathbf x\in\mathbb R^n$는 input vector다(이전 layer의 activation)
    - $\mathbf w\in\mathbb R^n$는 weight vector다(neuron으로 들어오는 edge의 weight)
    - $b\in\mathbb R$는 bias다
    - $f$는 activation function이다
    
    $$
    y=f\left(\sum_{i=1}^n w_i x_i+b\right)=f(\mathbf w^\top\mathbf x +b)
    $$
    
- 입력이 $n_\text{in}$개이고 neuron이 $n_\text{out}$개인 layer는 matrix multiplication 한 번으로 계산한다
    - 따라서 $\mathbf x\in\mathbb R^{n_\text{in}}$ (column vector)
    - 모든 weight vector를 weight matrix $W\in\mathbb R^{n_\text{out}\times n_\text{in}}$ 하나로 쌓는다
        - 각 row는 neuron 하나로 들어가는 weight다
    - bias는 vector $\mathbf b\in\mathbb R^{n_\text{out}}$로 쌓는다
    - 출력 hidden state의 shape은 $\mathbf h\in\mathbb R^{n_\text{out}}$다
    
    $$
    \mathbf h=f(W\mathbf x+\mathbf b)
    $$
    
- 실제로는 입력 $m$개를 batch로 한 번에 처리한다
    - 이때 입력을 matrix $X \in\mathbb R^{m\times n_\text{in}}$의 row로 배치한다
    - 관례상 $W$의 shape을 $\mathbb R^{n_\text{in}\times n_\text{out}}$로 바꾼다
        - 각 column이 neuron 하나로 들어가는 weight다
    - layer는 다음과 같다
        
        $$
        H=f(XW+\mathbf b)
        $$
        
        - $\mathbf b$는 shape $m\times n_\text{out}$으로 broadcast된다

> 수학 표기에서 linear layer는 $X\in\mathbb R^{m\times n_\text{in}}$를 받아 $W\in\mathbb R^{n_\text{in}\times n_\text{out}}$를 $XW+b$ 꼴로 적용한다.
>
> PyTorch에서는 weight matrix `W`를 실제로 $n_\text{out}\times n_\text{in}$로 저장한다. forward pass에서 `W`를 transpose해 `X @ W.T` $(m, n_\text{in})\times (n_\text{in},n_\text{out})$를 계산한다. transpose는 stride만 바꾸므로 비용이 없다. 이렇게 하면 $W$의 gradient가 자연스럽게 $n_\text{out}\times n_\text{in}$로 나와 $W$의 shape과 맞는다.

- $Z=XW+b$의 backprop을 구해 보자
    
    $$
    \frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top\quad (m,n_\text{out})\times (n_\text{out},n_\text{in})=(m,n_\text{in})\\[1em]
    \frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z}\quad (n_\text{in}, m)\times (m,n_\text{out})=(n_\text{in},n_\text{out})
    $$
    
    - 같은 bias $\mathbf b\in\mathbb R^{n_\text{out}}$가 모든 sample에 더해지고, sample마다 $\mathbf b$에 대한 gradient를 따로 만든다
        - 그래서 이 gradient들은 누적된다
        
        $$
        \frac{\partial L}{\partial b_j}=\sum_{i=1}^m\frac{\partial L}{\partial z_{ij}}\cdot\frac{\partial z_{ij}}{\partial b_j}=\sum_{i=1}^m\frac{\partial L}{\partial z_{ij}}\cdot 1=\sum_{i=1}^m\frac{\partial L}{\partial z_{ij}}
        $$
        
    - 가장 직관적인 이해 방법
        - $X$가 example 하나라면 $\partial L/\partial X$는 $\partial L/\partial Z \cdot W^\top (n_\text{in})$이다
        - $X$에 batch dimension이 있으면 출력 shape은 $(m,n_\text{in})$이어야 한다
        - $Z$의 row $i$는 $X$의 row $i$에만 의존한다(batch example끼리는 상호작용하지 않는다)
        - 따라서 row별 gradient를 그냥 쌓으면 된다
    - 일반적으로는 example 하나에 대한 Jacobian을 먼저 구한다(깔끔한 2차원이다)
        - tensor가 batch 전체에서 공유되면($W$처럼) batch dimension을 합해서 없앤다 → contract (batch dim이 inner dimension인 matmul)
        - tensor가 공유되지 않으면($X$, activation처럼) batch dimension을 유지한다 → stack (batch dim이 바깥에 있는 matmul)
    - PyTorch 구현처럼 $W\in\mathbb R^{n_\text{out}\times n_\text{in}}$이고 $Z=XW^\top$이면 다음과 같다
        
        $$
        \frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z} W\quad (m,n_\text{out})\times (n_\text{out}, n_\text{in})=(m,n_\text{in})\\[1em]
        \frac{\partial L}{\partial W}=\left(\frac{\partial L}{\partial Z}\right)^\top X\quad (n_\text{out}, m)\times (m,n_\text{in})=(n_\text{out},n_\text{in})
        $$
        

## Activation functions

- **sigmoid** $\sigma(x)\in(0,1)$
    
    $$
    \sigma(x)=\frac{1}{1+e^{-x}}
    $$
    
    - 출력을 확률로 해석하기 좋다
    - neural net의 hidden layer에는 쓰지 않는다
        - derivative가 $\sigma(x)(1-\sigma(x)) \leq 0.25$라서 vanishing gradient가 생긴다
        - zero-centered가 아니라서 한 node의 downstream gradient가 모두 양수이거나 모두 음수가 된다(upstream grad 부호에 따라)
- **tanh** $\in(-1,1)$
    
    $$
    \tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2\sigma(2x)-1
    $$
    
    - derivative는 $x=0$에서 최댓값 1.0이고, 여전히 vanish할 수 있다
    - $\tanh^\prime(z)=1-\tanh^2(x)\in(0,1]$이므로 $\tanh^\prime$ 항은 곱할수록 줄어들기만 한다
- **softmax** → 확률 분포
    
    $$
    \text{softmax}(\mathbf x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
    $$
    
    - temperature를 넣으면
        
        $$
        \text{softmax}(\mathbf x/T)_i=\frac{e^{x_i/T}}{\sum_j e^{x_j/T}}
        $$
        
- **ReLU** $\in(0, \infty)$
    
    $$
    \operatorname{ReLU}(x)=\max(x,0)
    $$
    
    - derivative는 $x>0$에서 1, $x<0$에서 0이다
    - dying ReLU: pre-activation이 영구히 음수가 되면(모든 입력에서 음수) gradient를 영원히 0으로 받는다
        - 학습 중 network 일부가 죽을 수 있다
- **Leaky ReLU** $\in(-\infty,\infty)$
    
    $$
    \text{LeakyReLU}(x)=\begin{cases}x&\text{if }x>0\\\alpha x&\text{if }x\leq0\end{cases}
    $$
    
    - dying ReLU 문제를 해결한다
- **Swish** (smooth, non-monotonic)
    
    $$
    \text{Swish}(x)=x\cdot\sigma(x)
    $$
    
- **GLU**는 linear projection 하나로 "content"를 만들고[왼쪽], 다른 하나로 gate를 만든다[오른쪽]
    
    $$
    \text{GLU}(x)=xW_1\odot\sigma(xW_2)
    $$
    
- **SwiGLU**는 GLU 안의 activation function 자리에 Swish를 넣는다
    
    $$
    \text{SwiGLU}(x)=(x W_1)\odot \text{Swish}(xW_2)
    $$
    

- non-linearity가 없으면 neural net은 linear transform 이상을 할 수 없다
    - layer를 더 쌓아도 linear transform 하나로 합쳐진다 $W_1 W_2x=Wx$
    - non-linearity 없이 layer를 늘려도 표현력은 늘지 않는다
    - non-linearity를 포함한 layer를 쌓으면 어떤 복잡한 함수든 근사할 수 있다

## Gradients

- 변수에 대한 derivative는 그 값이 바뀔 때 전체 식이 얼마나 민감하게 변하는지 알려준다
    - $\partial f/\partial x=3$이면 $x$를 작은 $h$만큼 바꿀 때 $f(x)$는 약 $\sim 3h$만큼 바뀐다
    
    $$
    \frac{df(x)}{dx}=\frac{f(x+h)-f(x)}{h}
    $$
    
- gradient $\nabla f$는 partial derivative를 모은 vector다
- 출력이 $m$개, 입력이 $n$개인 함수의 **Jacobian**은 partial derivative로 이뤄진 $m\times n$ matrix다
    
    $$
    f(\mathbf{x})=[f_1(x_1,...,x_n),...,f_m(x_1,...,x_n)]\\[1em]
    \frac{\partial f}{\partial x}=\begin{bmatrix}\frac{\partial f_1}{\partial x_1}&\cdots &\frac{\partial f_1}{\partial x_n}\\
    \vdots&\ddots&\vdots\\
    \frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}\end{bmatrix}
    $$
    
- 입력이 $n$개이고 출력이 scalar인 함수의 **Hessian**은 second partial derivative로 이뤄진 $n\times n$ matrix이고, $H_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}$다
    - loss function의 Hessian은 loss landscape의 curvature를 알려준다
- chain rule
    - 변수 하나짜리 함수의 합성에서는 derivative를 곱한다
        
        $$
        x=3y, y=x^2\\[1em]
        \frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}=3\cdot 2x=6x
        $$
        
    - 변수가 여러 개인 함수에서는 Jacobian을 곱한다
        
        $$
        \mathbf h=f(\mathbf z), \mathbf z=\mathbf W\mathbf x+\mathbf b\\[1em]
        \frac{\partial\mathbf h}{\partial\mathbf x}=\frac{\partial\mathbf h}{\partial\mathbf z}\frac{\partial\mathbf z}{\partial\mathbf x}=\cdots
        $$
        

- neural network 설정
    
    $$
    \begin{align*}\mathbf x&\in\mathbb{R}^d\\
    \mathbf h&=f(\mathbf W\mathbf x+\mathbf b)\in\mathbb{R}^k & \mathbf W\in\mathbb{R}^{k\times d},\mathbf b\in\mathbb{R}^k\\
    \mathbf s&=\mathbf u^\intercal\mathbf h\in\mathbb{R} & \mathbf u\in\mathbb{R}^k\end{align*}
    $$
    

- element-wise activation function $\mathbf h=f(\mathbf z)$에서 $\mathbf h,\mathbf z\in\mathbb{R}^n$일 때 ${\partial\mathbf h}/{\partial\mathbf z}$는 무엇인가?
    
    $$
    \left(\frac{\partial \mathbf h}{\partial \mathbf z}\right)_{ij}=\frac{\partial h_i}{\partial z_j}=\frac{\partial}{\partial z_j}f(z_i)=\begin{cases}f'(z_i)&\text{if }i=j\\0&\text{otherwise}\end{cases}
    $$
    
    - Jacobian은 diagonal matrix다
    
    $$
    \frac{\partial\mathbf h}{\partial \mathbf z}=\begin{bmatrix}f'(z_1)&&\\
    &\ddots&\\
    &&f'(z_n)\end{bmatrix}=\operatorname{diag}(f'(\mathbf z))
    $$
    
- 자주 쓰는 Jacobian
    
    $$
    \frac{\partial}{\partial \mathbf x}(\mathbf W\mathbf x+\mathbf b)=\begin{bmatrix}\ddots&&\\&\frac{\partial z_i}{\partial x_j}\\&&\ddots\end{bmatrix}=\begin{bmatrix}\ddots&&\\&W_{ij}\\&&\ddots\end{bmatrix}=\mathbf W\\[1em]
    \frac{\partial}{\partial \mathbf b}(\mathbf W\mathbf x+\mathbf b)=\mathbf I\\[1em]
    \frac{\partial}{\partial \mathbf z} f(\mathbf z)=\operatorname{diag}(f'(\mathbf z))\\[1em]
    \frac{\partial}{\partial \mathbf u}(\mathbf u^\intercal\mathbf h)=\mathbf h^\intercal
    $$
    
- 그 밖에 유용한 derivative
    
    $$
    \begin{align*}
    \frac{d}{dx} \frac 1x &=-\frac{1}{x^2}\\
    \frac{d}{dx} e^x &= e^x\\
    \frac{d}{dx} \sigma(x) &= (1-\sigma(x))\sigma(x)\quad\text{[using quotient rule]}\\
    \frac{d}{dx}\log x &= \frac 1x\\
    \frac{d}{dx}\tanh(x) &= 1-\tanh^2(x)
    \end{align*}
    $$
    
    - sigmoid의 derivative 구하기
        
        $$
        \begin{align*}
        \frac{d}{dx}\sigma(x) &= \frac{d}{dx}(1+e^{-x})^{-1} \\
        &=(1+e^{-x})^{-2}\cdot e^{-x}&\text{chain rule}\\
        &= \frac{e^{-x}}{(1+e^{-x})^2}\\
        &= \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}\\
        &=\sigma(x)\cdot(1-\sigma(x))
        \end{align*}
        $$
        
- Swish의 derivative
    
    $$
    \begin{align*}
    \frac{\partial}{\partial x}\text{Swish}(x)&=\frac{\partial}{\partial x}x\cdot\sigma(x)\\
    &=\sigma(x)+x\cdot\sigma'(x)&\text{product rule}\\
    &=\sigma(x)+x\cdot\sigma(x)(1-\sigma(x))&\text{derivative of sigmoid}\\
    &=\sigma(x)+x\sigma(x)-x\sigma(x)^2\\
    &=\sigma(x)+\text{Swish}(x)(1-\sigma(x))
    \end{align*}
    $$
    
- softmax + CE loss의 gradient
    - $\mathbf z\in\mathbb R^{\mathcal V}$를 $i\in\{1,…,\mathcal V\}$로 index하는 logits, $\mathbf p\in\mathbb R^{\mathcal V}$를 softmax를 거친 확률이라 하자
    
    $$
    p_i=\frac{e^{z_i}}{\sum_je^{z_j}}
    $$
    
    - CE loss gradient $\partial L/\partial \mathbf p$
        - $L=-\log p_t$이고, $t$는 정답 class다
        - gradient는 다음과 같다
            
            $$
            \frac{\partial L}{\partial p_i}=\begin{cases}-\frac{1}{p_t}&\text{if }i=t\\0&\text{otherwise}\end{cases}
            $$
            
            $$
            \frac{\partial L}{\partial \mathbf p}=\begin{bmatrix}0&\cdots&-\frac{1}{p_t}&\cdots &0\end{bmatrix}\in\mathbb R^{\mathcal V}
            $$
            
        - chain rule로 $\partial L/\partial\mathbf z$를 $\partial \mathbf p/\partial\mathbf z$로 표현한다
            - $\frac{\partial L}{\partial p_i}$는 $i=t$일 때만 0이 아니므로 대부분의 항이 사라진다
            
            $$
            \frac{\partial L}{\partial z_i}=\sum_{j}\frac{\partial L}{\partial p_j}\frac{\partial p_j}{\partial z_i}=-\frac{1}{p_t}\frac{\partial p_t}{\partial z_i}
            $$
            
            $$
            \frac{\partial L}{\partial\mathbf z}=\frac{\partial L}{\partial\mathbf p}\frac{\partial\mathbf p}{\partial\mathbf z}=\begin{bmatrix}0&\cdots&-\frac{1}{p_t}&\cdots &0\end{bmatrix}\begin{bmatrix}\frac{\partial p_1}{\partial z_1}&\cdots&\frac{\partial p_1}{\partial z_{\mathcal V}}\\&\ddots&\\\frac{\partial p_{\mathcal V}}{\partial z_1}&\cdots&\frac{\partial p_{\mathcal V}}{\partial z_{\mathcal V}}\end{bmatrix}=-\frac{1}{p_t}\frac{\partial p_t}{\partial z_i}\in\mathbb R^{\mathcal V}
            $$
            
    - 이제 softmax gradient $\partial \mathbf p/\partial\mathbf z$를 구한다
        
        $$
        \frac{\partial\mathbf p}{\partial\mathbf z}=\begin{bmatrix}\frac{\partial p_1}{\partial z_1}&\cdots&\frac{\partial p_1}{\partial z_{\mathcal V}}\\&\ddots&\\\frac{\partial p_{\mathcal V}}{\partial z_1}&\cdots&\frac{\partial p_{\mathcal V}}{\partial z_{\mathcal V}}\end{bmatrix}\in\mathbb R^{\mathcal V\times\mathcal V}
        $$
        
        $$
        \frac{\partial p_j}{\partial z_i}=
        \begin{cases}p_j(1-p_j)&\text{if }i=j\\
        -p_jp_i&\text{if }i\neq j
        \end{cases}
        $$
        
    - 모두 합쳐 $\partial L/\partial\mathbf z$를 구한다
        - 정답 token $(i=t)$이면
        
        $$
        \frac{\partial L}{\partial z_t}=-\frac{1}{p_t}\frac{\partial p_t}{\partial z_t}=-\frac{1}{p_t}\cdot p_t(1-p_t)=p_t-1
        $$
        
        - 나머지 token이면
        
        $$
        \frac{\partial L}{\partial z_i}=-\frac{1}{p_t}\cdot(-p_tp_i)=p_i
        $$
        
    - 결과가 아주 깔끔하다: $\partial L/\partial\mathbf z = \mathbf p - \operatorname{one\_hot}(t)$

## Backpropagation

- NN 식은 computation graph로 표현한다
    - 함수의 어느 부분을 "gate"로 볼지는 편의의 문제다. 보통 local gradient가 쉬운 부분을 gate로 잡는다
- **backpropagation**은 gate들이 gradient signal로 서로 소통하는 과정으로 볼 수 있다. loss를 줄이려면 각 gate의 출력이 커져야 하는지 작아져야 하는지, 얼마나 강하게 바뀌어야 하는지를 전달한다
    - chain rule을 반복 적용해 이뤄진다. chain rule 덕분에 각 gradient를 upstream gradient(이미 계산됨)와 local gradient로 나눌 수 있다
- graph의 각 node는 upstream gradient를 받아 downstream gradient를 내려보낸다
    - 각 node에는 local gradient(입력에 대한 출력의 gradient)가 있다
    - downstream gradient = upstream gradient $\times$ local gradient

- 바깥으로 갈라지는 branch에서는 gradient를 더한다
    - $y$가 $a$와 $b$ 계산에 모두 쓰이면 $\frac{\partial f}{\partial y}=\frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}$
- node별 직관
    - $+$는 upstream gradient를 각 항에 그대로 나눠준다
    - $\max$는 upstream gradient를 여러 입력 중 하나로 "route"한다
    - $\times$는 downstream gradient에서 forward 계수를 서로 바꾼다
- backprop
    - output gradient를 1로 초기화한다
    - reverse topological order로 node를 방문한다. 각 node의 gradient는 후속 node의 gradient로 계산한다
    - 제대로 구현하면 forward prop과 backprop의 big-$O$ 복잡도는 같다
- automatic differentiation
    - gradient 계산은 forward prop의 symbolic expression에서 자동으로 유도할 수 있다
    - 각 node type은 출력 계산법과, 출력에 대한 gradient가 주어졌을 때 입력에 대한 gradient 계산법을 알아야 한다
        - local gradient는 프로그래머가 작성한다
- 수동 gradient checking
    - 모든 parameter $x$에 대해 $x-h$와 $x+h$에서 $f$를 다시 계산하고 다음을 확인한다
        
        $$
        f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}
        $$
        
- backward pass에는 중간 activation이 필요하다
    - 그래서 NN은 보통 forward pass 동안 모든 중간 activation을 저장한다
    - **activation / gradient checkpointing**은 activation 일부("checkpoint")만 저장해 memory를 아끼고 compute를 더 쓴다. 저장하지 않은 activation이 필요하면 가장 가까운 checkpoint에서 partial forward pass를 돌려 그때그때 다시 계산한다
        - layer가 N개인 model을 K개 segment로 checkpoint하면
            - memory는 $O(N)$에서 $O(K+N/K)$가 된다
            - backward compute는 $O(N)$에서 $O(N + N*(K-1)/K)$가 된다
            - 최적은 $K=\sqrt{N}$이다: memory $O(\sqrt{N})$, backward compute $\sim O(2N)$
- 연습해 볼 것
    - $\partial f/\partial x$를 명시적인 식으로 구하면 매우 복잡하지만, 그럴 필요가 전혀 없다
    - forward pass에서 중간 변수를 여러 개 만든다. 각각은 local gradient를 아는 단순한 식이다
    
    $$
    f(x,y)=\frac{x+\sigma (y)}{\sigma(x)+(x+y)^2}
    $$
    
- backprop은 scalar에서 시작해야 한다. backprop은 모든 parameter $\theta$에 대해 $\partial L/\partial\theta$를 구하고, 이는 parameter마다 숫자 하나다. $L$이 scalar일 때만 말이 된다
    - scalar에서 `.backward()`를 호출하면 PyTorch는 backward pass를 $\partial L/\partial L=1$로 암묵적으로 시작한다
    - token별 loss $\ell_1,…,\ell_n$이 있고 $L=\frac 1N\sum_i\ell_i$로 정의하면(mean reduction), derivative의 linearity에 따라 다음이 성립한다
        
        $$
        \frac{\partial L}{\partial\theta}=\frac 1N\sum_{i=1}^N\frac{\partial\ell_i}{\partial\theta}
        $$
        
        - 따라서 mean loss의 gradient는 각 $\ell_i$를 따로 backprop해서 얻은 gradient의 평균과 정확히 같다
- upstream gradient는 항상 activation에 대한 gradient다. parameter에 대한 gradient는 update에 쓰이고 거기서 끝난다. parameter는 computation graph의 leaf node이기 때문이다
    
    ```python
    loss
      │  dL/dy2 (activation grad)
      ▼
    Layer 2 ──→ dL/dW2 (param grad, stored)
      │  dL/dy1 (activation grad)
      ▼
    Layer 1 ──→ dL/dW1 (param grad, stored)
      │  dL/dx  (activation grad — usually discarded)
      ▼
    input
    ```
    

## Optimizers

- vanilla SGD update
    
    $$
    \theta\leftarrow \theta-\eta g_t
    $$
    
- **optimizer**는 parameter update의 방향과 크기를 정한다
- Adam은 parameter tensor마다 다음을 들고 있다
    1. parameter 자체 ($\theta$)
    2. gradient ($g$)
    3. first moment (momentum)
    4. second moment (variance)
- **Adam optimizer**
    
    $$
    \theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
    $$
    
    - $g$를 현재 step의 gradient라 하자
    - first moment $m$은 gradient의 running mean이다
        
        $$
        m\leftarrow\beta_1 m+(1-\beta_1)g
        $$
        
        - 결과: gradient가 계속 한 방향을 가리켜 왔다면 그 방향으로 더 확신 있게 움직인다
            - 시간에 따라 velocity가 쌓여 noisy gradient를 뚫고 나가는 데 도움이 된다
    - second moment $v$는 gradient 제곱의 running mean이다
        
        $$
        v\leftarrow\beta_2 v+(1-\beta_2)g^2
        $$
        
        - 결과: weight마다 사실상 다른 learning rate를 갖는다. gradient가 계속 크면 step이 작아진다
            - gradient를 같은 scale로 normalize하는 효과가 있다
    - bias correction: 학습 초기에 running average가 초기값 쪽으로 치우친 것을 보정한다
        - time step $t$가 1부터 시작할 때의 식이다(아니면 $t+1$을 쓴다)
        
        $$
        \hat m_t=\frac{m_t}{1-\beta_1^t},\quad\hat v_t=\frac{v_t}{1-\beta_2^t}
        $$
        
    - hyperparameter $\beta_1$과 $\beta_2$가 moment estimate의 update를 조절한다
    - $m$과 $v$는 모두 $0$으로 초기화한다
    - 그래서 parameter당 memory는 parameter 크기의 약 4배다
- **AdamW**는 Adam에 0을 향하는 **weight decay**를 더한다
    
    $$
    \theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}-\eta\lambda \theta
    $$
    
- optimizer는 model parameter와 `lr`로 초기화한다. parameter는 optimizer가 최적화할 값을 알려주고, `lr`는 update 크기를 정한다
- LR schedule로 할지 optimizer로 할지 어떻게 정하나?
    - time step $t$에만 의존하면 → 아마 LR schedule
    - parameter별 history가 필요하면 → optimizer
- 코드는 다음과 같다
    - `params`로 parameter group을 만든다. group마다 hyperparameter를 따로 둘 수 있다(예: layer마다 다른 learning rate)
        - `torch.optim.AdamW(model.parameters())`는 parameter group 하나를 만든다
        - 보통 bias와 LayerNorm parameter에는 weight decay를 주지 않는다
            
            ```python
            torch.optim.AdamW([
                {'params': decay_params, 'weight_decay': 0.01},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ])
            ```
            
    - `defaults` dict는 parameter group에 명시하지 않은 hyperparameter의 기본값을 준다
    - weight decay는 parameter에 의존하므로 실제로는 Adam update 전에 적용하는 편이 낫다

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:  # for every group of parameters
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:  # for every parameter in the group
                if p.grad is None:
                    continue
                state = self.state[p]

                # state initialization with 0s
                t = state.get("t", 0)
                m, v = state.get("m", torch.zeros_like(p.data)), state.get("v", torch.zeros_like(p.data))

                # weight decay
                p.data -= lr * weight_decay * p.data

                # Adam update
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))
                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                # update optimizer state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
```

- **gradient clipping**은 grad norm의 크기를 제한한다
    - *모든* gradient의 global norm을 계산한다
    - 최댓값을 넘으면 모든 parameter를 같은 비율로 줄여 최댓값 아래로 맞춘다
    - 한 step이 파국적으로 커지는 것을 막는다

## Learning rate

- warmup은 학습 초기 example의 primacy effect를 줄인다


---

# Mathy things

## Information theory

- **cross entropy**

    $$
    \operatorname{CE}(p,q)=-\mathbb E_p[\log q]=-\sum_{x\in\mathcal X}p(x)\log q(x)
    $$

- **KL divergence**

    $$
    \operatorname{KL}(p\mid\mid q)=\sum_{x\in\mathcal X}p(x)(\log p(x)-\log q(x))
    $$

- **entropy**

    $$
    H(p)=-\sum_{x\in\mathcal X}p(x)\log p(x)
    $$

    - logits $x_i$가 주어졌을 때 자주 쓰는 다른 형태

        $$
        H(p)=\log\sum_ie^{x_i}-\underbrace{\frac{\sum_ie^{x_i}x_i}{\sum_i e^{x_i}}}_{\mathbb E[x]}
        $$

- $p$와 $q$ 사이의 cross entropy는 $p$와 $q$ 사이의 KL에 $p$의 줄일 수 없는 entropy를 더한 값이다

    $$
    \operatorname{CE}(p,q)=\operatorname{KL}(p\mid\mid q)+H(p)
    $$

    - 증명

        $$
        \begin{align*}
        \operatorname{KL}(p\mid\mid q)&=\sum_{x\in\mathcal X}p(x)(\log p(x)-\log q(x))\\
        &=\sum_{x\in\mathcal X}p(x)\log p(x)-\sum_{x\in\mathcal X}p(x)\log q(x)\\
        &=-H(p)+\operatorname{CE}(p,q)
        \end{align*}
        $$

- **cross-entropy loss**
    - target 분포가 1-hot이면 cross-entropy loss는 다음 token의 negative log likelihood다

    $$
    \mathcal L(x)=-\sum_{t=1}^T\log p(x_t\mid x_{<t})
    $$

    - KL divergence와도 같다
    - loss 구현
        - `F.cross_entropy()`를 쓰면 logits와 labels를 내부에서 shift한다
            - `loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=pad_idx)`

    ```python
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
    # build loss mask
    masked_logprobs = -token_logprobs * mask.float()
    return masked_logprobs.sum() / mask.sum()
    ```

## Numerical stability and other tricks

- 일반적으로 다음을 조심한다
    - 큰 x에서 exp(x) → $\infty$로 overflow
    - 0에 가까운 x에서 log(x) → $-\infty$로 underflow ($\log(0)=-\infty$)
    - 1에 가까운 x에서 log(x) → 정밀도 문제 ($\log(1) = 0$)
- `softmax(x)` 계산
    - 큰 $x_i$에서 $\exp(x_i)$가 overflow하므로 불안정하다
    - *softmax는 상수를 빼도 값이 그대로*라는 성질을 쓴다

        $$
        \begin{align*}
        \text{softmax}(x)_i &= \frac{e^{x_i}}{\sum_j e^{x_j}}\\
        &= \frac{e^{x_i-c}\cdot e^c}{\sum_j e^{x_j-c}\cdot e^c}\\
        &= \frac{e^{x_i-c}\cdot e^c}{e^c\cdot \sum_j e^{x_j-c}}\\
        &= \frac{e^{x_i-c}}{\sum_j e^{x_j-c}}\\
        &=\text{softmax}(x-c)_i
        \end{align*}
        $$

    - 모든 $x_i$에서 $x_\text{max}$를 빼면 $x_i$가 커지지 않는다. 가장 큰 지수가 $\exp(0)=1$이 된다. 수치적으로 안정한 구현은 이렇게 한다

        $$
        \text{softmax}(x)_i=\frac{e^{x_i-x_\text{max}}}{\sum_j e^{x_j-x_\text{max}}}
        $$

- `log(softmax(x))` 계산
    - log-softmax를 그대로 계산하면 안 된다. 0에 가까운 입력, 즉 확률이 낮은 class의 log가 불안정하다
    - 대신 `x - logsumexp(x)`를 쓴다. 아주 작은 확률을 직접 만들지 않고, `logsumexp(x)`는 안정하다
    - 두 식은 같다

    $$
    \begin{align*}
    \log(\text{softmax}(x))_i &= \log \frac{e^{x_i}}{\sum_j e^{x_j}}\\
    &= \log e^{x_i}-\log\sum_j e^{x_j}\\
    &= x_i-\log\sum_j e^{x_j}
    \end{align*}
    $$

- `log(sum(exp(x)))` 계산
    - 왜 불안정한가?
        - $x_i$ 하나라도 크면 $\exp(x_i)$가 무한대로 overflow한다
            - 예: float32에서는 $x_i\approx 83$이면 overflow한다
        - 모든 $x_i$가 아주 큰 음수면 $\log(0)$이 되어 $-\infty$로 underflow한다
        - 합이 1에 가까우면 $\log(1)$ 근처가 불안정해서 정밀도 문제가 생긴다
            - distillation 프로젝트에서 겪은 문제다
    - 직관: $x_i$ 값을 작게 만들고 싶다. 상수를 빼면 된다! 마지막에 그 상수를 다시 더하면 된다
    - `logsumexp(x)`는 이렇게 구현한다

        $$
        \begin{align*}
        \log\sum_i e^{x_i}&=\log\sum_i\left(e^{x_i-x_\text{max}} \cdot e^{x_\text{max}}\right)\\
        &= \log \left(e^{x_\text{max}}\sum_i e^{x_i-x_\text{max}}\right)\\
        &=\log e^{x_\text{max}}+\log\sum_ie^{x_i-x_\text{max}}\\
        &= x_\text{max}+\log\sum_i e^{x_i-x_\text{max}}
        \end{align*}
        $$

        - 가장 큰 항은 $x_i=x_\text{max}$일 때 $e^0=1$이라 overflow가 없다
        - 합이 최소 1이라 $\log(0)$을 계산할 일이 없다
- softmax를 단순하게 계산하면 pass가 두 번 필요하다. 안정성을 위한 $x_\text{max}$를 구하는 pass와 분모 $\sum_j e^{x_j-x_\text{max}}$를 구하는 pass다
    - stable softmax

        $$
        \text{softmax}(x)_i=\frac{e^{x_i-x_\text{max}}}{\sum_j e^{x_j-x_\text{max}}}
        $$

    - **online softmax trick**: $x_\text{max}$와 분모 $\sum_j e^{x_j-x_\text{max}}$ 계산을 한 pass로 합친다
        - 아이디어: 지금까지의 max로 분모를 계산하고, max가 새로 바뀔 때마다 분모를 rescale한다
        - 다음 두 값을 유지한다
            - running maximum $m_k=\max(x_1,...,x_k)$
            - running (shifted) denominator $d_k=\sum_j e^{x_j-m_k}$
        - update rule: $x_{k+1}$이 들어오면

            $$
            m_{k+1}\leftarrow\max(m_k,x_{k+1})\\
            d_{k+1}\leftarrow d_k\cdot e^{m_k-m_{k+1}}+e^{x_{k+1}-m_{k+1}}
            $$

        - $d_{k+1}$ update가 맞는 이유

            $$
            \begin{align*}
            d_{k+1}&=\sum_{j=1}^{k+1}e^{x_j-m_{k+1}}\\
            &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_{k+1}}&\text{split off last term}\\
            &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_k+m_k-m_{k+1}}&\text{algebraic manipulation}\\
            &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_k}e^{m_k-m_{k+1}}\\
            &= e^{x_{k+1}-m_{k+1}}+e^{m_k-m_{k+1}}\underbrace{\sum_{j=1}^k e^{x_j-m_k}}_{d_k}&\text{pull out a constant}\\
            &= \underbrace{d_k\cdot e^{m_k-m_{k+1}}}_\text{rescaling of prev terms}+\underbrace{e^{x_{k+1}-m_{k+1}}}_\text{new term}&\text{express in terms of }d_k\text{, reorder}
            \end{align*}
            $$

            - max가 바뀌지 않으면 $m_k=m_{k+1}$이고 rescaling factor $e^{m_k-m_{k+1}}=1$이다
        - softmax 자체를 반환하려면 logits를 한 번 더 돌면서 $e^{x_i-m_S}/d_S$를 계산해야 한다
- 같은 방법으로 logits $x_i$와 values $v_i$가 stream으로 들어올 때 weighted sum도 계산할 수 있다

    $$
    o=\sum_ip_iv_i=\sum_i\frac{e^{x_i}}{\sum_j e^{x_j}}v_i=\frac{\sum_i e^{x_i}v_i}{\sum_i e^{x_i}}
    $$

    - FlashAttention에서 $o$는 각 query $q$의 attention output이다. $x_i=q\cdot k_i$이고 $v_i$는 value vector다
    - 수치적으로 안정한 버전은 $e^{x_i}$의 overflow를 피하려고 분자와 분모에 모두 $e^{-x_\text{max}}$를 곱한다

        $$
        o=\frac{\sum_i e^{x_i-x_\text{max}}v_i}{\sum_i e^{x_i-x_\text{max}}}
        $$

    - $m_k$, $d_k$와 함께 running numerator $o_k\in\mathbb R^H$도 유지한다. $H$는 각 $v_i$의 차원이고 FlashAttention에서는 head dimension이다

        $$
        o_k=\sum_{i=1}^ke^{x_i-m_k}v_i
        $$

        - update는 $d_k$ update와 같은 아이디어다. 유도 과정도 $d_k$와 같다

            $$
            o_k=o_k\cdot e^{m_k-m_{k+1}}+ \underbrace{e^{x_{k+1}-m_{k+1}}v_{k+1}}_\text{new term}
            $$

    - logits $N$개를 모두 처리하면 $(m_N, d_N, o_N)$이 남고, 실제 attention output은 $o_N/d_N$이다

> $e^{x}$가 들어간 식을 수치적으로 안정하게 만들 때는 $e^{-m}$을 곱하고 무엇이 남는지 보는 게 표준 방법이다. $m$은 큰 수이고 보통 $m=x_\text{max}$로 잡는다.

## Basic statistics

- $p$-value: null hypothesis가 참일 때 적어도 이만큼 극단적인 데이터를 관측할 확률
    - 핵심은 null이 주어졌을 때 데이터의 확률이라는 점이다. 데이터가 주어졌을 때 null의 확률이 아니다
- 이 그룹들은 서로 다른가?
    - **Kolmogorov-Smirnov test**: 연속 변수를 관측한 두 그룹이 같은 분포에서 나왔는가?
        - 두 sample의 CDF 사이 최대 수직 거리를 잰다
    - **Chi-squared test**: 범주형 변수를 관측한 두 그룹이 같은 분포에서 나왔는가?
    - **T-test**: 연속형 관측값 두 그룹의 평균이 다른가?
        - 가정: 데이터가 정규분포를 따른다
        - one-sample 버전: 한 그룹의 평균이 특정 값과 같은가
        - two-sample 버전: 두 그룹의 평균이 같은가
        - paired 버전: 같은 항목을 두 번 잰 값이 체계적으로 다른가
            - 사실상 one-sample t-test다
            - 예제별 차이가 +1, 0, -1 중 하나로 주어지면, 그 차이가 0과 유의하게 다른지 검정할 수 있다
    - **ANOVA** (F-test): t-test를 세 그룹 이상으로 일반화한 것이다. $k$개 그룹 중 평균이 다른 그룹이 있는가?
    - **McNemar’s tests**: 같은 dataset에서 두 classifier를 비교한다
        - 두 모델을 같은 test set에서 평가하고 각 예제가 맞음 또는 틀림으로 갈리는 표준 설정에서는 아마 이 test가 가장 낫다
- 이 변수들은 관련이 있는가?
    - **Pearson correlation test**: 두 연속 변수 사이에 선형 관계가 있는지 검정한다
        - correlation coefficient $r$과, $r$이 0과 유의하게 다른지에 대한 p-value를 함께 준다
        - 비선형 관계는 전혀 잡지 못한다. 완벽한 포물선 관계도 $r\approx 0$이다
        - scatterplot에 직선을 맞춘다고 생각하면 된다
    - **Spearman correlation test**: 같은 아이디어지만 단조 관계를 잰다
        - 두 변수를 rank로 바꾸고, 가장 작은 값이 rank 1이 된다. 그다음 rank에 대해 Pearson correlation을 계산한다
        - 구체적인 값이 아니라 *순서*만 중요하다
    - Pearson versus Spearman
        - Pearson은 outlier에 민감하고 Spearman은 outlier에 강하다
        - Pearson은 비선형 단조 관계를 과소평가한다
        - Pearson이 해석이 더 깔끔하다
    - mutual information: 비선형을 포함해 두 변수 사이의 모든 의존성을 잡는다
        - 엄밀히는 통계 검정이 아니다

## Gradient flow through sampling

- **Gumbel-Max trick**
    - logits $z_1,…,z_k$가 있으면 다음 방법으로 해당 categorical distribution에서 sample할 수 있다
        1. $\operatorname{Gumbel}(0,1)$ 분포에서 독립적인 noise $g_1,…,g_k$를 뽑는다
        2. $(z_1+g_1,…,z_k+g_k)$에서 argmax를 취한다
- **Gumbel-Softmax**는 argmax를 softmax로 바꾼다: $\operatorname{softmax}((z_1+g_1,…,z_k+g_k)/\tau)$
    - softmax는 모든 곳에서 미분 가능하다!
    - temperature가 높으면 gradient가 부드럽고, 낮으면 sample이 discrete해진다
    - 미분 가능하게 만드는 게 목적이 아니다. plain softmax도 이미 미분 가능하다. stochastic하게 만드는 게 목적이다
    - plain softmax: $y = \operatorname{softmax}(\alpha)$
        - deterministic하다. 항상 같은 soft mixture가 나온다
    - true categorical sampling: $y=\operatorname{one\_hot}(\operatorname{sample}(\operatorname{softmax}(\alpha)))$
    - gumbel-softmax: $y=\operatorname{softmax}((\alpha+G)/\tau)$
        - stochastic하다 (Gumbel noise)
        - 거의 discrete하다 (낮은 temperature)
        - 미분 가능하다
        - exploration과 gradient를 둘 다 얻는다!
- **straight-through estimator**: $f$가 identity function인 척한다
    - forward: $y=f(x)$ (미분 불가능한 함수를 적용한다)
    - backward: $\frac{\partial\mathcal L}{\partial x}=\frac{\partial\mathcal L}{\partial y}$
        - upstream gradient를 그대로 downstream gradient로 넘긴다
    - $f(x)$가 감소 함수면 틀린다. 신호 방향이 반대가 된다
    - 하지만 STE는 보통 단조 증가하는 연산에 쓴다 (rounding, quantization, 증가하는 step function)
        - 크기는 틀려도 방향은 맞다

## Theoretical CS

- **regular language**는 **finite state machine**으로 인식할 수 있는 언어다. finite state machine은 **finite automaton**이라고도 한다
    - deterministic finite automaton (DFA)
- **context-free language**는 pushdown automaton으로 인식할 수 있는 언어다. pushdown automaton은 사실상 finite state machine에 stack을 더한 것이다
    - stack은 메모리 제한이 없지만 맨 위로만 접근할 수 있다
    - 프로그래밍 언어 문법은 context-free language로 만든다 (짝이 맞는 괄호, 중첩된 함수 호출, 짝이 맞는 HTML tag 등)
- 유한 언어는 당연히 regular하다. state를 충분히 두면 유효한 문자열을 모두 나열할 수 있다
- 모든 DFA는 ReLU RNN으로 표현할 수 있다
    - 다음과 같은 DFA가 있다고 하자
        - states $Q=\{q_1,…,q_k\}$
        - alphabet $\Sigma=\{\sigma_1,…,\sigma_m\}$
        - transition function $\delta:Q\times\Sigma\to Q$
        - start state $q_1$
        - accept state $F\subseteq Q$
    - hidden dimension이 $k$인 RNN을 만든다. hidden state는 state의 one-hot encoding이다
    - $W_h\in\mathbb R^{k\times k}$, $W_x\in\mathbb R^{k\times m}$, $b\in\mathbb R^k$를 만들어야 한다
    - 각 transition $\delta(q_i,\sigma_k)=q_j$마다 $(W_h)_{ji}=1$, $(W_x)_{jk}=1$, $b_j=-1$로 둔다

---

# The modern transformer LM

## Architecture

| **symbol** | **dimension** |
| --- | --- |
| B | batch의 sequence 수 |
| L | layer 수 |
| T | sequence length (생성할 token 수) |
| S | sequence length (주어진 context) |
| V | vocab size |
| D | hidden dimension |
| H | head dimension |
| F | MLP hidden dimension, 보통 F = 4D |
| N | query head 수, N * H = D |
| K | key/value head 수, GQA에서는 K < N |
| G | GQA의 group size = N // K |

**token embedding**

- embedding matrix $\mathbf W_e\in\mathbb R^{V\times D}$, 초기 hidden states $\mathbf X^{(0)}\in\mathbb R^{B\times S\times D}$

$$
\mathbf X^{(0)}=\mathbf {W}_e[\text{tokens}]
$$

**layer loop (for** $\ell\in[0,…,L-1]$**)**

- RMSNorm은 $\mathbf X^{(\ell)}$의 모든 원소를 $\mathbf X^{(\ell)}$의 RMS로 나눠 hidden state의 RMS를 1로 만든다. 그다음 학습되는 rescaling parameter $\gamma$를 곱한다

    $$
    \bar{\mathbf X}^{(\ell)}=\frac{\mathbf X^{(\ell)}}{\text{RMS}(\mathbf X^{(\ell)})+\epsilon}\odot\gamma_{\text{attn}}^{(\ell)}\\[1em]
    \operatorname{RMS}(\mathbf X)=\sqrt{\frac 1D\sum_{i=1}^D x_i^2}
    $$

- 각 head는 $W^{(\ell)}_Q\in\mathbb R^{D\times D}$, $W^K\in\mathbb R^{D\times KH}$, $W^{(\ell)}_V\in\mathbb R^{D\times KH}$로 $\mathbf {X}^{(\ell)}$를 head의 저차원 subspace에 projection한다. 여기서 $H=D/N$이다

    $$
    \mathbf Q=\bar {\mathbf X}\mathbf W_Q\in\mathbb R^{B\times T\times D},\quad\mathbf K=\bar {\mathbf X}\mathbf W_K\in\mathbb R^{B\times S\times D},\quad\mathbf V=\bar {\mathbf X}\mathbf W_V\in\mathbb R^{B\times S\times D}
    $$

    - [선택] QK norm: query와 key vector에 RMSNorm을 적용해 dot product에 들어가는 vector 크기를 조절한다
- reshape로 head 차원을 드러낸다. $D\to N\times H$, $K\cdot H\to K\times H$로 바꾼 뒤 sequence length 차원($S$ 또는 $T$)과 head 차원($N$ 또는 $K$)을 transpose한다

    $$
    \mathbf Q\in\mathbb R^{B\times T\times D}\to \mathbb R^{B\times N\times T\times H}\\
    \mathbf K\in\mathbb R^{B\times S\times (K\cdot H)}\to \mathbb R^{B\times K\times S\times H}\\
    \mathbf V\in\mathbb R^{B\times S\times (K\cdot H)}\to \mathbb R^{B\times K\times S\times H}
    $$

- GQA를 위해 $K$, $V$를 expand한다

    $$
    \mathbf K\in\mathbb R^{B\times K\times S\times H}\to \mathbb R^{B\times N\times S\times H}\\
    \mathbf V\in\mathbb R^{B\times K\times S\times H}\to\mathbb R^{B\times N\times S\times H}
    $$

- 각 position $m$에서 query vector $\mathbf q_m\in\mathbf R^H$ 또는 key vector $\mathbf k_m$을 $\mathbf R_m$으로 회전해 RoPE를 적용한다
    - dimension pair $i$, 즉 $\mathbf q_m$의 index $(2i, 2i+1)$는 각도 $m\theta_i$만큼 회전한다. 여기서 $\theta_i=\Theta^{-\frac{2i}{H}}$다
    - hyperparameter $\Theta$는 기본 회전 frequency를 정하고 $H$는 head dimension이다

        $$
        \mathbf R_m=\begin{bmatrix}
        \ddots&&\\
        &\mathbf R_m^{(i)}&\\
        &&\ddots\\
        \end{bmatrix}\in\mathbb R^{H\times H}\quad\text{where}\quad \mathbf R_m^{(i)}=\begin{bmatrix}
        \cos(m\theta_i)&-\sin(m\theta_i)\\
        \sin(m\theta_i)&\cos(m\theta_i)\\
        \end{bmatrix}
        $$


    $$
    \mathbf q_m\leftarrow \mathbf R_m \mathbf q_m, \quad\mathbf k_m\leftarrow \mathbf R_m \mathbf k_m, 
    $$

- attention score를 계산한다
    - head dimension $H$로 나눈다. 그러지 않으면 dot product가 $\sqrt H$에 비례해 커진다
        - softmax 입력이 크면 → 분포가 뾰족해진다 → update가 잘 안 된다

        $$
        \mathbf A=\frac{\mathbf Q\mathbf K^\top}{\sqrt H}\in\mathbb R^{B\times N\times T\times S}
        $$

    - causal mask를 적용한다

        $$
        \mathbf A_{ij}\leftarrow \begin{cases} \mathbf A_{ij}&\text{if }j\leq i\\
        -\infty&\text{if }j>i
        \end{cases}
        $$

    - softmax를 적용한다

        $$
        \mathbf A=\operatorname{softmax}(\mathbf A)=\frac{\exp(\mathbf A_{ij})}{\sum_{k=1}^S \exp(\mathbf A_{ik})}
        $$

- value의 weighted sum으로 attention output을 얻는다

    $$
    \mathbf O=\mathbf A\mathbf V\in\mathbb R^{B\times N\times T\times H}
    $$

- reshape

    $$
    \mathbf O\in\mathbb R^{B\times N\times T\times H}\to\mathbb R^{B\times T\times D}
    $$

- output projection $W^{(\ell)}_O\in\mathbb R^{D\times D}$를 적용해 여러 head의 출력을 섞는다

    $$
    \mathbf O_\text{proj}=\mathbf O\mathbf W_O
    $$

- residual connection

    $$
    \mathbf X^{(\ell)} \leftarrow\mathbf X^{(\ell)}+\mathbf O_\text{proj}
    $$


**feed forward network**

- RMSNorm

    $$
    \bar{\mathbf X}^{(\ell)}=\frac{\mathbf X^{(\ell)}}{\text{RMS}(\mathbf X^{(\ell)}) + \epsilon}\odot\gamma_{\text{ffn}}^{(\ell)}
    $$

- $\mathbf W^{(\ell)}_\text{up}\in\mathbb R^{D\times F}$, $\mathbf W^{(\ell)}_\text{gate}\in\mathbb R^{D\times F}$로 gate, up projection을 한다 [확장]

    $$
    \mathbf U=\bar {\mathbf X}\mathbf W_\text{up}\\
    \mathbf G=\bar {\mathbf X}\mathbf W_\text{gate}
    $$

- SwiGLU activation

    $$
    \operatorname{Swish}(\mathbf G)=\mathbf G\odot\sigma(\mathbf G)=\mathbf G\odot\frac{1}{1+e^{-\mathbf G}}\\[1em]
    \mathbf H=\operatorname{Swish}(\mathbf G)\odot\mathbf U\in\mathbb R^{B\times T\times F}
    $$

- $\mathbf W_\text{down}^{(\ell)}\in\mathbb R^{F\times D}$로 down projection을 한다

    $$
    \mathbf F=\mathbf H\mathbf W_\text{down}
    $$

- residual connection

    $$
    \mathbf X^{(\ell+1)} =\mathbf X^{(\ell)}+\mathbf F
    $$


**final layer norm**

- final norm

    $$
    \mathbf X_\text{final}=\frac{\mathbf X^{(L)}}{\text{RMS}(\mathbf X^{(L)}) + \epsilon}\odot\gamma_{\text{final}}
    $$


**unembedding**

- $\mathbf W_u\in\mathbb R^{D\times V}$로 vocab 차원에 projection한다

    $$
    \mathbf Z=\mathbf X_\text{final}\mathbf W_u\in\mathbb R^{B\times T\times V}
    $$


## Implementation notes

- softmax 전 attention score를 만들 때 `scores.masked_fill(~mask, -torch.inf)`를 쓴다
    - `mask`가 attend *할 수 있는* 위치에서 `True`라는 규칙을 가정한다
    - `tensor.masked_fill(mask, value)`는 `mask`가 `True`인 곳의 `tensor` 값을 `value`로 채운다
- RoPE
    - 모든 (position, index) 쌍 $(m,i)$에 대해 $\cos(m\theta_i)$와 $\sin(m\theta_i)$를 cache한다
        - 초기화할 때 해 두면 된다

        ```python
        positions = torch.arange(max_seq_len, device=device)  # shape (max_seq_len)
        thetas = self.theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)  # shape (d_k // 2)
        angles = positions.unsqueeze(-1) * thetas.unsqueeze(0)
        ```

    - 실제로는 2x2 matmul을 여러 번 하지 않고 회전을 dot product로 표현한다
        - 마지막 head dimension $H$를 $(H/2,2)$로 reshape해 $\mathbf Q$, $\mathbf K$의 짝수, 홀수 index를 뽑는다

            ```python
            x_pairs = x.reshape(*x.shape[:-1], -1, 2)
            x_even = x_pairs[..., 0]
            x_odd = x_pairs[..., 1]
            ```

        - 회전된 행렬의 짝수, 홀수 위치를 모두 계산한다

            ```python
            x_out_even = x_even * cos - x_odd * sin
            x_out_odd = x_even * sin + x_odd * cos
            ```

        - 둘을 나란히 stack한 뒤 flatten해서 interleave한다
            - `torch.stack()`은 새 차원을 추가한다

            ```python
            torch.stack([x_out_even, x_out_odd], dim=-1).flatten(start_dim=-2)
            ```

- attention은 이렇게 생겼다
    - `d_model`($D$)을 `num_heads x head_dim`($N\times H$)으로 펼치려면 `.reshape()`가 필요하다
    - queries, keys, vectors를 나누려면 `qkv.unbind()`가 필요하다 [구현에 따라 선택]
    - attention 계산을 위해 `num_heads`와 `seq_len` 차원을 바꾸려면 `.transpose()`가 필요하다
    - `output`을 얻은 뒤 원래 shape로 돌리려면 다시 `.transpose()`와 `.reshape()`를 한다

    ```python
    batch_size, seq_len, _ = x.shape
    
    x_norm = self.norm(x)
    
    qkv = self.qkv_proj(x_norm)  # (batch, seq_len, 3 * d_model)
    
    qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)  # (batch, seq_len, num_heads, head_dim)
    q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    output = scaled_dot_product_attention(q, k, v, mask)  # (batch, num_heads, seq_len, head_dim)
    
    output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
    output = output.reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)
    output = self.out_proj(output)  # (batch, seq_len, d_model)
    
    return x + output
    ```

    - attention 부분은 이렇다

        ```python
        def scaled_dot_product_attention(q, k, v, mask):
            """
            k, q: (batch_size, ..., seq_len, d_k)
            v: (batch_size, ..., seq_len, d_v)
            returns o (batch_size, ..., seq_len, d_v)
            """
            d_k = q.shape[-1]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
            scores = scores.masked_fill(~mask, -torch.inf)
            return softmax(scores, dim=-1) @ v
        ```


## Accounting

### Model parameters

- embedding: $(V,D)$
- attention은 $2D^2+2DKH\approx 4D^2$다 (standard multi-head attention에서 $N=K$일 때)
    - $Q$는 $(D, D)$
    - $K$는 $(D,KH)$
    - $V$는 $(D,KH)$
    - $O$는 $(D,D)$
- FFN은 $3DF$다
    - up projection은 $(D,F)$
    - gate projection은 $(D,F)$
    - down projection은 $(F,D)$
- layer norm은 layer마다 $2D$다 (final norm은 따로 더한다)
    - pre-attention, pre-FFN layernorm이 각각 parameter $D$개를 가진다. $D$의 각 차원마다 $\gamma$가 하나다
- unembedding: $(V,D)$
- 합계: $2VD+L(4D^2+2D+3DF)\approx 2VD+12LD^2$ ($F=8D/3$일 때)
    - 전체 모델 parameter 수는 $2VD+12LD^2$다

### Model activations

- attention activations: $6BSD+BNS^2$
    - layer norm 입력은 $(B,S,D)$
    - layer norm 출력은 $(B,S,D)$
    - Q, K, V 출력은 $(B,S,D)$, $(B,S,KH)$, $(B,S,KH)$
    - attention score는 $(B,N,S, S)$
    - attention output은 $(B,S,D)$
- FFN activations: $2BSD+2BSF\approx 8BSD$ ($F=8/3D$일 때)
    - layer norm 입력은 $(B,S,D)$
    - gate, up projection 출력은 각각 $(B,S,F)$
    - down projection 출력은 $(B,S,D)$
- layer당 activations는 $14BSD+BNS^2$다

### FLOPs in forward pass

- prefill 단계를 가정한다 (그래서 $S=T$)
- attention은 layer당 $8BSD^2+4BS^2D$다
    - $Q$ projection은 $(B,S,D)\times(D,D)$ → $2BSD^2$ FLOPs
    - $K$ projection은 $(B,S,D)\times (D,KH)$ → $2BSDKH\approx 2BSD^2$ ($K=N$일 때)
    - $V$ projection은 $(B,S,D)\times (D,KH)$ → $2BSDKH\approx 2BSD^2$ ($K=N$일 때)
    - $QK^\top$는 $(B,N,S,H)\times(B,N,H,S)$ → $2BNS^2H=2BS^2D$ ($D=NH$이므로)
    - $AV$는 $(B,N,S,S)\times (B,N,S,H)$ → $2BS^2D$
    - $O$ projection은 $(B,S,D)\times(D,D)$ → $2BSD^2$ FLOPs
- FFN은 layer당 $6BSDF\approx16BSD^2$다 ($F=8D/3$일 때)
    - up projection은 $(B,S,D)\times (D,F)$ → $2BSDF$
    - gate projection은 $(B,S,D)\times (D,F)$ → $2BSDF$
    - down projection은 $(B,S,F)\times (F,D)$ → $2BSDF$
- layer당 합계: $8BSD^2+4BS^2D+16BSD^2=2BSD(12D+2S)$
- unembedding layer는 $2BSDV$다
    - unembedding은 $(B,S,D)\times (D,V)$ → $2BSDV$
- 전체 forward pass는 $2LBSD(12D+2S)+2BSDV\approx 2BSD(12LD+2LS+V)$다

### FLOPs in backward pass

- 보통 forward pass FLOPs의 2$\times$로 잡는다
    - parameter에 대한 gradient와 input에 대한 gradient를 각각 matmul 한 번으로 계산한다
        - input에 대한 gradient $\partial L/\partial X$는 이전 layer로 들어가는 gradient다

### Inference memory use

- inference 때 전체 메모리 사용량: model weights + KV cache + peak activations
    - 모델 parameter 수: $2VD + 12LD^2$

        ```python
        num_params = sum(p.numel() for p in model.parameters())
        ```

    - KV cache 크기: $B\cdot S\cdot (KH)\cdot L\cdot 2$
        - $B$ = batch size
        - $S$ = sequence length
        - $K$ = KV head 수
        - $H$ = head dimension
        - $L$ = layer 수
        - K와 V 두 개라서 2
    - activations: prefill에서 $O(BNS^2 + BSF)$, decode에서 $O(BNS+BF)$
        - `torch.inference_mode()`는 메모리를 바로 해제하므로 layer 하나의 peak memory만 따지면 된다
        - 입력은 prefill에서 $B\times S\times D$, decode에서 $B\times T\times D$
        - FlashAttention을 쓰면 $S\times S$ 행렬을 만들지 않는다 → attention이 대신 $O(S)$가 된다
        - peak activations
            - $B\cdot S\cdot D$ = layer 입력
            - $B\cdot S\cdot 3\cdot D$ = K, Q, V vector
            - $B\cdot N \cdot S^2$ = attention matrix (FlashAttention 없이)
                - batch의 예제마다, query head마다 $S\times S$ 행렬이 하나씩 있다
            - $B\cdot S\cdot F$ = FFN 중간값
        - activations는 FA 없이는 sequence length $S$의 제곱에, FA를 쓰면 $S$에 선형으로 비례한다
- batch size와 sequence length가 작으면 weights가 대부분을 차지한다
- prefill 단계에서
    - sequence length가 길면 (큰 $S$) activation의 $S^2$ attention 항이 대부분을 차지한다
- batch size가 크면 (큰 $B$) KV cache와 activations가 모두 커진다

### Train memory use

- 학습 때 전체 메모리 사용량: model weights + optimizer states + gradients + activations
    - 모델 parameter $P$개
        - FP32 master weights [full 또는 mixed precision] → $4P$
        - forward pass용 BF16 임시 복사본 [mixed precision] → $2P$
    - optimizer states $2P$개 (first moment, second moment)
        - FP32 Adam states [mixed precision] → $8P$
    - gradient $P$개
        - FP32 → $4P$
            - mixed precision에서도 gradient는 BF16으로 계산하지만 FP32로 accumulate한다
    - activations
        - 보통 가장 큰 부분이다. $B$, $S$, $D$, $L$에 따라 달라진다
        - layer당 activations는 $14BSD+BNS^2$다 (flash attention 없이)
            - flash attention을 쓰면 두 번째 항이 $BNS$가 되고, activation 메모리는 전체 token 수인 $BS$에 비례한다
        - backward pass에서 gradient를 계산할 때 필요하다. gradient checkpointing으로 줄일 수 있다

## Attention

- standard attention은 $O(n^2)$다
- KV cache가 sequence length와 무관해진다
    - token이 window 밖으로 나가면 그냥 버리면 된다
- **sliding window**: 각 token은 마지막 $W$개 token에만 attend한다. 그래서 계산량이 $O(n^2)$가 아니라 $O(nW)$다
    - token $n$개가 각각 $O(W)$ 일을 한다 ($W$개 key/value에 attend)
- **sparse attention**
- local attention과 global attention을 번갈아 쓴다

## RMSNorm

- normalization은 exploding/vanishing gradient를 막아 학습을 안정시킨다
    - 모든 차원을 똑같이 scaling하면 제약이 너무 크다. feature마다 필요한 크기가 다를 수 있다
    - $\gamma$로 normalization의 안정성과 차원별 크기 차이를 함께 얻는다
- $\gamma\in\mathbb R^D$는 학습되는 차원별 rescaling이다
    - normalization 단계는 hidden state의 RMS를 강제로 1로 만들어 학습된 scale 정보를 없앤다
- $\gamma$는 network에 feature별 크기 조절 능력을 돌려준다
    - $\gamma_i>1$ → 차원 $i$를 키운다
    - $\gamma_i<1$ → 차원 $i$를 줄인다
    - $\gamma_i\approx 0$ → 차원 $i$를 없앤다

## SwiGLU FFN

- $\mathbf G$와 $\mathbf U$ 모두 content에 기여한다
    - $\mathbf U$는 학습된 representation 하나를 주고, $\mathbf G$는 또 다른 representation을 준다. $\mathbf G$는 자기 confidence로 self-gate된다

## RoPE

- RoPE는 query와 key vector만 회전하고 value는 회전하지 않는다
    - 위치 정보는 어떤 token끼리 attend하는지에만 영향을 주면 된다. 어떤 정보가 전달되는지에는 영향을 줄 필요가 없다
- position $m$의 query vector $\mathbf q$와 position $n$의 key vector $\mathbf k$에 대해, dot product $\mathbf q\cdot\mathbf k$가 *상대* 위치 $m-n$에만 의존하길 원한다
    - dot product $\langle f(\mathbf q,m),f(\mathbf k, n)\rangle$가 상대 위치 형태로만 정보를 담는 함수 $g$가 되도록 하는 $f$를 찾는다. 예:

        $$
        \langle f(\mathbf q,m),f(\mathbf k, n)\rangle=g(\mathbf q,\mathbf k,n-m)
        $$

    - RoPE embedding이 그런 해 중 하나다

        $$
        f(\mathbf x,m)=R_{m\theta}\mathbf x\\
        g(\mathbf q,\mathbf k,n-m)=\mathbf q^\top R_{(n-m)\theta}\mathbf k
        $$

        - 증명

            $$
            \begin{align*}
            \langle R_{m\theta}\mathbf q,R_{n\theta}\mathbf k\rangle&=(R_{m\theta}\mathbf q)^\top R_{n\theta}\mathbf k\\
            &=\mathbf q^\top R_{m\theta}^\top R_{n\theta}\mathbf k\\
            &=\mathbf q^\top R_{-m\theta}R_{n\theta}\mathbf k
            &\text{because }R_\alpha^\top=R_{-\alpha}\\
            &=\mathbf q^\top R_{(n-m)\theta}\mathbf k&\text{because } R_\alpha R_\beta=R_{\alpha+\beta}
            \end{align*}
            $$

- 거리가 멀어질수록 inner product가 감소한다
- 2D vector $\mathbf x=[x_1,x_2]$를 각도 $\theta$만큼 회전하면

    $$
    R_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
    $$

    - position $m$과 head dimension index $i$에서는 $m\theta_i$만큼 회전한다

        $$
        R_{m\theta}\mathbf x=\begin{bmatrix}
        x_1\cos(m\theta_i)-x_2\sin(m\theta_i)\\
        x_2\sin(m\theta_i)+x_2\cos(m\theta_i)\\
        \end{bmatrix}
        $$

- $H$차원 embedding은 $H/2$개 pair로 나누고, pair마다 다른 frequency로 독립적으로 회전한다
    - $\Theta$는 RoPE의 유일한 hyperparameter다. 보통 10,000이고 HF에서는 `rotary_base`라고 부른다
        - 모델이 기본적으로 구분할 수 있는 가장 긴 거리를 정한다
        - $m\theta_i=2\pi$일 때 한 바퀴를 돈다 ⇒ $m=\frac{d\pi}{\theta_i}$
            - 가장 느린 (가장 작은) $\theta_i$는 $\Theta^{-1}$이다 → $m=d\pi\Theta$
            - 그래서 가장 느린 pair는 $2\pi\Theta$ position에 걸쳐 한 바퀴를 돈다
    - dimension pair $i$마다 $\theta_i$는

        $$
        \theta_i=\Theta^{-2i/H}
        $$

        - 이렇게 하면 간격이 지수적으로 벌어진다. original transformer의 sinusoidal positional encoding과 같은 설계다
            - frequency가 log 스케일로 고르게 퍼진다 → 여러 scale을 효과적으로 덮는다
        - low-frequency pair (큰 $i$ → 작은 $\theta_i$)는 천천히 돈다 → 인접 position 사이 변화가 거의 없다 → 장거리 정보를 담는다
        - high-frequency pair (작은 $i$ → 큰 $\theta_i$)는 빨리 돈다 → 인접 position끼리 크게 다르다 → 가까운 위치를 구분할 수 있다
- position 하나에 대한 전체 rotation matrix는 block-diagonal이다. $2\times 2$ block 하나가 차원 pair 하나를 맡는다

    $$
    \mathbf R_m=\begin{bmatrix}
    \ddots&&\\
    &\mathbf R_m^{(i)}&\\
    &&\ddots\\
    \end{bmatrix}
    $$

- 복소수로 다시 쓸 수 있다
    - 각 pair $[x_{2i},x_{2i+1}]$를 복소수 $z=x_{2i}+ix_{2i+1}$로 본다
    - 각도 $\theta$만큼 회전하는 것은 $e^{i\theta}$를 곱하는 것과 같다
        - Euler’s theorem: $e^{i\theta}=\cos\theta+i\sin\theta$

            $$
            \begin{align*}
            z e^{i\theta}&=(a+bi)(\cos\theta+i\sin\theta)&\text{Euler's Theorem}\\
            &=(a\cos\theta-b\sin\theta)+i(a\sin\theta+b\cos\theta)
            \end{align*}
            $$

        - 복소평면에서 rotation matrix를 적용하는 것과 같다

            $$
            \begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\begin{bmatrix}a\\b\end{bmatrix}
            $$

- 실제로는 전체 rotation matrix를 만들지 않고 element-wise 곱을 한다

    $$
    \begin{bmatrix} x_1\\x_2\end{bmatrix}\odot
    \begin{bmatrix}
    \cos(m\theta)\\
    \cos(m\theta)
    \end{bmatrix}+\begin{bmatrix} -x_2\\x_1\end{bmatrix}\odot
    \begin{bmatrix}\sin(m\theta)\\\sin(m\theta)\end{bmatrix}
    $$


---

# Inference

- **latency**는 요청 하나를 끝내는 데 걸리는 시간이다. 단위는 초다.
- **throughput**은 모든 요청을 통틀어 단위 시간당 처리하는 토큰(또는 요청) 수다. 단위는 tokens/second다.

## Batching & packing

- **traditional batching**: 요청 $N$개를 모아 함께 처리하고, 모두 끝날 때까지 기다린 뒤 다음 batch를 모은다.
    - 한 시퀀스는 토큰 500개를 만들고 다른 시퀀스는 10개만 만들면, 짧은 쪽은 할 일 없이 기다린다.
- **continuous batching**: 시퀀스 하나가 끝나는 즉시 그 자리에 새 요청을 넣는다. batch 전체를 기다리지 않는다.
    - batch가 늘 꽉 찬다.
- **selective batching**: prefill 단계 시퀀스와 generation 단계 시퀀스를 영리하게 섞는다.
    - 아이디어: prefill은 compute-heavy이고 decode는 memory-bound다.
- **sequence packing**: 예제들을 max sequence length까지 이어 붙이고, attention mask로 서로 섞이지 않게 막는다.
- **token-budget batching**: batch 안의 총 토큰 수(그 batch의 최대 seqlen으로 padding한 뒤 기준)가 정해진 budget을 넘지 않게 batch를 짠다.
    - 주로 finetuning에서 쓴다.
    - GPU 메모리가 `batch_size x sequence_length`로 정해지니 합리적이다.

## Speculative decoding

- **speculative decoding**은 prefill이 generation보다 빠르다는 점을 이용한다.
- draft model $q$로 토큰 $K$개를 생성한다.
- target model $p$로 그 토큰들을 평가한다.
    - 각 draft 토큰 $x$를 다음 확률로 accept한다.
        
        $$
        \min\left(1,\frac{p(x)}{q(x)}\right)
        $$
        
        - $p(x)>q(x)$면 반드시 받는다.
    - reject되면 조정된 분포 $\max(0, p(x)-q(x))$를 renormalize해서 샘플링한다.
        - 처음 reject된 토큰부터 시작한다.
    - draft 토큰을 채점할 때 다음 토큰 logits를 "공짜로" 얻으므로, 토큰 하나는 항상 teacher에서 샘플링한다.
- P(토큰 $x$ 출력) = P(draft $x$) * P(accept $x$) + P(샘플링된 토큰 reject) * P(sample $x$)
    - case 1 (첫째 항): draft의 $x$가 accept된 경우
        
        $$
        q(x)\cdot\min\left(1,\frac{p(x)}{q(x)}\right)=\min(q(x),p(x))
        $$
        
    - case 2 (둘째 항): reject 뒤에 $x$가 뽑힌 경우
        - draft 토큰 $x'$를 뽑고 reject할 확률:
            - $p(x')>q(x')$면 0
            - 그 밖에는
                
                $$
                q(x')\cdot\left(1-\frac{p(x')}{q(x')}\right)=q(x')-p(x')
                $$
                
        - 따라서 전체 reject 확률은 다음과 같다.
            
            $$
            \begin{align*}
            P(\text{rejection})&=\sum_{x'}\max(0,q(x')-p(x'))\\
            &=\sum_{x'}\max(0,p(x')-q(x'))
            \end{align*}
            $$
            
            - 둘째 줄은 $p$와 $q$ 모두 합이 1이라서 성립한다.
        - reject하면 다음 분포에서 샘플링한다.
            
            $$
            \frac{\max(0,p(x)-q(x))}{\sum_{x'}\max(0,p(x')-q(x'))}
            $$
            
        - 따라서 reject 뒤에 $x$를 얻을 확률은 다음과 같다.
            
            $$
            \sum_{x'}\max(0,p(x')-q(x'))\cdot \frac{\max(0,p(x)-q(x))}{\sum_{x'}\max(0,p(x')-q(x'))}=\max(0,p(x)-q(x))
            $$
            
    - 두 경우를 합치면 다음과 같다.
        
        $$
        \min(q(x),p(x))+\max(0,p(x)-q(x))=p(x)
        $$
        

## KV cache

- cache를 쓰는 단일 토큰 forward pass
    - cache 차원은 `(batch, num_heads, max_seq_len, head_dim)`이어야 한다.
    - 실제로는 self attention 모듈 안에 `self.kv_cache`로 cache를 둔다.
    
    ```python
    # Pre-allocate cache based on max_seq_len
    kv_cache = [
        {
            'k': torch.zeros(batch, num_heads, max_seq_len, head_dim, device='cuda', dtype=torch.float16),
            'v': torch.zeros(batch, num_heads, max_seq_len, head_dim, device='cuda', dtype=torch.float16),
        }
        for _ in range(num_layers)
    ]
    
    # For attention, use cache[:, :, :position+1, :] as keys/values
    
    def forward_with_cache(model, new_token, kv_cache, position):
        """
        Instead of processing full sequence, process just the new token
        and reuse cached K, V from previous positions.
        """
        # Embed just the new token
        x = model.embed(new_token)  # (batch, 1, d_model)
        
        for layer_idx, layer in enumerate(model.layers):
            q, k, v = layer.qkv_proj(x).chunk(3, dim=-1)
            
            # Update cache
            kv_cache[layer_idx]['k'][:, :, position, :] = k.squeeze(2)
            kv_cache[layer_idx]['v'][:, :, position, :] = v.squeeze(2)
            
            # Attend over all cached positions
            k_full = kv_cache[layer_idx]['k'][:, :, :position+1, :]
            v_full = kv_cache[layer_idx]['v'][:, :, :position+1, :]
            
            x = attention(q, k_full, v_full)
            x = layer.ffn(x)
        
        return model.lm_head(x)
    ```
    

### Reducing KV cache size

**lower dimension of KV cache**

- 표준 **multi-head attention** (MHA) transformer attention에서 KV cache는 `num_layers × num_heads × seq_len × head_dim × 2 (K and V)`로 커진다.
- **multi-query attention** (MQA)에서는 모든 head가 같은 K, V를 공유하고, Q만 head마다 따로 둔다.
    - KV cache가 `num_heads`배 줄어든다.
    - inference가 훨씬 빠르고 메모리도 덜 쓴다.
- **group-query attention** (GQA)에서는 head를 그룹으로 나누고, 그룹마다 K, V를 공유한다.
    - MHA와 MQA의 중간이다.
- MQA와 GQA는 head끼리 KV를 공유하므로 head마다 표현력이 조금 줄어든다.
- **multi-head latent attention** (MLA)은 DeepSeek v2에서 나왔다.
    - KV cache에 shape `(seq_len, num_heads × head_dim)`인 key, value를 두지 않고, 더 작은 latent vector `(seq_len, latent_dim)`를 cache한다.
    - decoding 단계마다 latent KV를 원래 크기로 다시 projection한다.
- MLA는 head마다 KV를 따로 두지만, 공유 latent space로 압축한다.
    - inference 때 계산이 조금 늘어난다 (latent에서 KV로 가는 projection).
    - MLA는 RoPE와 호환되지 않는다.

- **cross-layer attention**은 layer끼리 KV를 공유한다.

**local attention**

- 표준 attention은 $O(n^2)$다. local attention을 쓰면 KV cache가 sequence length와 무관해진다.
    - window 밖으로 나간 토큰은 버리면 된다.

## Sampling strategies

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)
    
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        
        indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## Flash Attention

- 표준 attention
    - 메모리 문제는 `attn_weights`가 `(batch, num_heads, seq_len, seq_len)`이라는 데 있다.
        - `seq_len` = 8192, head 32개, FP16이면 8192² × 32 × 2 bytes ≈ 4GB다.
    
    ```python
    import torch
    import torch.nn.functional as F
    import math
    
    def standard_attention(q, k, v):
        # q, k, v are all (batch, num_heads, seq_len, head_dim)
        
        scale = math.sqrt(q.size(-1))
        
        # Materialize full N×N attention matrix
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        return output
    ```
    
    - 표준 attention이 memory bound인 이유: $N\times N$ attention 행렬을 HBM에 쓰고, softmax를 위해 다시 읽고, 또 쓰고, 실제로 쓰려고 또 읽는다.
- **flash attention**은 online softmax trick으로 attention을 block 단위로 계산한다. 전체 attention 행렬을 GPU 메인 메모리에 쓰지 않고 중간 결과를 빠른 SRAM에 둔다.
    - attention의 activation 메모리를 $O(n^2)$에서 $O(n)$으로 줄인다.
    - 메모리 대역폭이 병목이 아니라서 빠르다.
    - inference 때는 `model.to_bettertransformer()`를 쓰거나 `from_pretrained()`에 `attn_implementation="flash_attention_2"`를 넘겨 켠다.
    
    ```python
    def flash_attention(q, k, v):
        # Same input shapes: (batch, num_heads, seq_len, head_dim)
        
        # PyTorch picks the best backend automatically (Flash Attention, memory-efficient, or math)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return output
    ```
    
- FlashAttention은 $N\times N$ 행렬 전체를 만들지 않는다. SRAM에 들어갈 만큼 작은 tile 단위로 attention을 계산한다.
- 개념적으로는 이렇다.
    1. $Q$의 block 하나를 불러온다 (예: 64행).
    2. K, V의 block 하나를 불러온다 (예: 64열).
    3. 그 tile의 attention score를 계산하고, softmax를 적용하고, V를 곱한다. 모두 SRAM에서 한다.
    4. 최종 출력만 HBM에 쓴다.
    5. 모든 tile에 반복한다.
- (batch, head) 조합마다 완전히 독립이므로, `batch × num_heads`개의 병렬 worker가 Q, K, V의 `m  × head_dim` tile을 각각 처리한다. `m`은 tile 크기다.
- *Flash Attention은 근사가 아니라 정확한 방법이다.*
- Flash Attention 쓰는 법
    - `is_causal=True` 플래그는 mask 행렬을 만들지 않고 kernel 안에 효율적으로 합쳐진다.
    
    ```python
    def flash_attention(q, k, v):
        # Same input shapes: (batch, num_heads, seq_len, head_dim)
        
        # PyTorch picks the best backend automatically (Flash Attention, memory-efficient, or math)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return output
    ```
    

---

# Scaling laws

- maximal update parameterization ($\mu P$)
    - 핵심 문제: 작은 규모에서 찾은 hyperparameter가 큰 규모로 일반화되지 않는다.
    - standard parameterization에서는 width가 바뀌면 layer마다 update 크기가 제각각이 된다.
    - $\mu P$는 layer별 initialization과 learning rate를 조정해서, weight 대비 update의 *크기*가 width와 상관없이 일정하게 한다.
    - 주로 width scaling을 조정한다.
- learning rate를 compute에 fit한다.
    
    $$
    \text{LR}(C)=\beta C^{-\alpha}\implies\log\text{LR}(C)=\log\beta-\alpha\log C
    $$
    
- loss를 compute에 fit한다.
    - irreducible loss 항 $\mathcal L_\infty$ (데이터의 entropy)가 필요하다. 없으면 $C\to\infty$일 때 $\mathcal L\to 0$이 된다.
    
    $$
    \mathcal L(C)=\mathcal L_\infty+\beta C^{-\alpha}\implies \log(\mathcal L(C)-\mathcal L_\infty)=\log\beta-\alpha\log C
    $$
    
- 식은 어떻게 fit하나? least squares를 쓴다.
    - least squares는 residual 제곱합을 최소화하는 objective다.
        - ordinary (또는 linear) least squares와 non-linear least squares가 있다.
    - linear least squares는 closed-form 해가 있다.
        - $y=X\beta$의 closed-form 해는 $\beta=(X^\top X)^{-1}X^\top y$다.
    - non-linear least squares는 반복해서 다듬어 푼다.
        
        $$
        S=\sum_i(y_i-f(x_i))^2
        $$
        

---

# GPUs

- **high bandwidth memory** (HBM)는 GPU의 메인 메모리다.
    - GPU 입장에서는 느린 메모리다.
    - A100은 40GB 또는 80GB다.
- **static RAM** (SRAM)은 작고 빠른 on-chip 메모리다.
    - A100 전체로 ~20MB다.

---

# Other architectures

## RNNs

### Vanilla RNN

- 각 step $t$에서 $x\in\mathbb R^D$와 이전 hidden state $h_{t-1}\in\mathbb R^H$를 처리해 다음 hidden state $h_t$를 만든다 ($D$는 input size, $H$는 hidden size).
    - weight 행렬은 $W_x\in\mathbb R^{H\times D}$, $W_h\in\mathbb R^{H\times H}$다.
    
    $$
    h_t=\tanh(W_{x}x_t+W_{h}h_{t-1}+b)\\
    y_t=W_\text{out} h_t+b_\text{out}
    $$
    
    - weight는 모든 time step에서 공유한다.
- vanishing gradient가 생기는 이유
    - $z_t=W_xx_t+W_hh_{t-1}+b$라 하자.
    
    $$
    \frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(\tanh'(z_t)) W_h
    $$
    
    - $\tanh^\prime\in(0,1]$이라서 $\tanh^\prime$은 줄이기만 한다.
    - $W$를 반복해서 곱하면 gradient가 vanishing하거나 exploding한다.

### LSTM

- LSTM은 forget, input, output gate가 있는 cell state $c_t$를 쓴다.
    - 각 step의 입력: 이전 hidden state $h_{t-1}$, 이전 cell state $c_{t-1}$, 현재 입력 $x_t$
    - **forget gate**: cell state에서 무엇을 지울지 정한다.
    
    $$
    f_t=\sigma(W_f\cdot [h_{t-1}, x_t]+b_f)
    $$
    
    - **input gate**: 어떤 새 정보를 쓸지 정한다.
        
        $$
        i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i)
        $$
        
    - cell state
        
        $$
        \tilde c_t=\tanh(W_c\cdot[h_{t-1},x_t]+b_c)
        $$
        
    - cell state update: 이전 cell state $c_{t-1}$에서 일부 정보를 지우고, 새 cell state $\tilde c_t$에서 일부 정보를 더한다.
        
        $$
        c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
        $$
        
    - **output gate**: 새 hidden state로 무엇을 내보낼지 정한다.
        
        $$
        o_t=\sigma(W_o\cdot[h_{t-1},x_t]+b_o)
        $$
        
    - 새 hidden state
        
        $$
        h_t=o_t\odot\tanh(c_t)
        $$
        
- $f_t$가 1 근처에 머무는 동안 gradient는 vanishing 없이 여러 timestep을 거슬러 갈 수 있다.
- cell state는 메모리고, hidden state는 작업용 출력이다.
    - cell state는 고속도로를 따라 흐르며 elementwise 곱셈과 덧셈만 거친다. matmul도 non-linearity도 없다.
    - 정보는 gate를 통해서만 더하거나 지운다.
- hidden state $h_t=o_t\cdot \tanh(c_t)$는 cell state를 걸러서 본 것이다.
    - 두 가지 역할을 한다.
        - 바깥으로 내보내는 cell의 출력
        - 다음 step에서 cell이 자기 자신에게 던지는 query. $t+1$의 gate는 $c_t$가 아니라 $h_t$로 계산하기 때문이다.
- cell state와 hidden state를 분리한 것이 LSTM이 작동하는 이유다.
    - RNN은 벡터 하나로 장기 메모리와 현재 출력을 모두 맡기려 한다.
- GRU는 gate를 reset, update 두 개만 쓰는 단순화 버전이다.

### vs. transformers

- *먼* time step의 gradient는 *앞선* time step의 처리에 영향을 주지 못한다.
- RNN vs transformer
    - attention은 두 토큰 사이에 직접적인 $O(1)$ 경로를 준다. 토큰 사이 거리와 상관없다.
        - 각 토큰이 다른 모든 토큰을 직접 본다.
        - RNN에서는 토큰 1의 정보가 토큰 100에 닿으려면 중간 hidden state를 모두 살아서 거쳐야 한다.
        - long-range dependency에 유리하다.
    - attention은 모든 쌍의 관계를 한 번에 계산한다.

## State space models

- SSM의 기본 식
    - $u_k$는 1D 입력 신호다.
    - $x_k$는 $N$차원 latent state다.
    
    $$
    x_k=A x_{k-1}+B u_k
    $$
    
- SSM은 training 때 $O(n)$이고 병렬이며, inference 때는 $O(1)$이고 순차적이다.
    - 반면 transformer는 병렬이지만 attention이 $O(n^2)$이고, RNN은 $O(n)$이지만 순차적이다.
- Mamba는 $B$, $C$, $\Delta$를 입력의 함수로 만든다.
    
    $$
    B_k=f_B(u_k)\\
    C_k=f_C(u_k)\\
    \Delta_k=f_\Delta(u_k)
    $$
    
    - $B$로 정보를 선택적으로 받아들인다.
    - $C$로 state에서 선택적으로 읽는다.
    - $\Delta$로 timescale을 조절한다.

---

# Post-training

## policy gradients

- 표기
    - action $a_t\in\mathcal V$: time step $t$의 next-token
    - state $s_t$: time step $t$의 text prefix $(s_0,a_0,…,a_{t-1})$
    - $a_t\sim\pi_\theta(\cdot\mid s_t)$: LM policy
    - $s_0\sim p_0$: prompt 분포 start distribution에서 샘플링한 prompt $s_0$
    - $\tau$: trajectory(finite-horizon). rollout, episode라고도 부른다
    - $R(\tau)$: trajectory $\tau$에서 얻은 reward
- 목표
    
    $$
    \text{maximize }J(\theta)=\mathbb E_{\tau\sim\pi_\theta} [R(\tau)]
    $$
    
- gradient ascent로 풀 수 있다
    
    $$
    \theta_{k+1}=\theta_k+\alpha\nabla_\theta J(\theta_k)
    $$
    
- **vanilla policy gradient**(**REINFORCE**라고도 한다): objective의 gradient는 다음처럼 쓸 수 있다
    
    $$
    \nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t) R(\tau)\right]
    $$
    
    - SFT와 거의 같은 update다. 다만 데이터 $\tau$를 policy에서 샘플링하고, gradient에 $R(\tau)$로 가중치를 준다
        - $R(\tau)$가 양수면 $\tau$의 각 token $a_t$에 대해 $\log\pi_\theta(a_t\mid s_t)$를 키우는 방향으로 간다
            - 음수면 반대 방향으로 간다
        - $R(\tau)$의 크기가 클수록 step도 커진다
    - gradient 유도
        
        $$
        \begin{align*}
        \nabla_\theta J(\theta)&=\nabla_\theta\mathbb E_{\tau\sim\pi_\theta} [R(\tau)]\\
        &=\nabla_\theta\sum_\tau P(\tau\mid\theta)\,R(\tau)\\
        &=\sum_\tau\nabla_\theta P(\tau\mid\theta)\,R(\tau)\\
        &=\sum_\tau P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)&\text{the log derivative trick: }\nabla P=P\,\nabla \log P\\
        &=\mathbb E_{\tau\sim\pi_\theta}\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)\\
        &=\mathbb E_{\tau\sim\pi_\theta}\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau)
        \end{align*}
        $$
        
- 실제로는 시작 state $s_0^{(i)}\sim p_0$에서 policy $\pi_\theta$로 rollout $\tau^{(i)}$를 $N$개 샘플링해 $\nabla_\theta J(\theta)$를 추정한다
    
    $$
    \hat g=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})(R(\tau^{(i)}))
    $$
    
- 흔히 policy gradient loss라고 부르는 것은 `pg_loss.backward()`가 근사 policy gradient $\hat g$와 같은 gradient를 내도록 만든 스칼라 `pg_loss`일 뿐이다
    
    $$
    L(\theta)=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})(R(\tau^{(i)}))
    $$
    
    - 일반적인 의미의 loss가 아니다. $L(\theta)$는 policy가 얼마나 좋은지 알려주지 않고, 올바른 gradient를 만드는 장치일 뿐이다
    - 현재 policy로 샘플링한 데이터로 만들기 때문에 고정된 objective가 없다

**baselined policy gradient**

- vanilla policy gradient의 문제는 추정치의 variance가 매우 크다는 점이다
    - 쉬운 prompt라서 모든 response가 양의 reward를 받는다고 하자
        - baseline이 없으면 batch 안의 나쁜 response까지 모두 강화된다!
            - 학습 전체로 평균 내면 괜찮지만 update가 매우 noisy해진다
        - 예를 들어 *평균* reward를 baseline으로 쓰면 평균 아래 response는 강화되지 않는다
- **baselined policy gradient**: gradient 추정에서 $R(\tau)$에 baseline function $b(s_t)$를 뺀다
    
    $$
    \nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-b(s_t))\right]
    $$
    
    - $b(s_t)$가 state $s_t$에만 의존하고 $a_t$에는 의존하지 않으면 $\nabla J(\theta)$ 추정에 bias가 생기지 않는다
    - $R(\tau)-b(s_t)$가 작아지도록 $b(s_t)$가 $R(\tau)$와 상관관계가 있으면 좋다. 그러면 gradient noise가 줄어든다
    - 최적화하는 objective는 그대로다. objective의 gradient를 추정하는 방법만 바뀐다
- baselined policy gradient가 policy gradient의 unbiased estimate인 이유
    - baseline 항 $B$의 기댓값이 0임을 보인다
    
    $$
    B=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)\right]
    $$
    
    - $X_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)$로 두고 다시 쓴다
        
        $$
        B=\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^T X_t
        $$
        
    - 먼저 기댓값을 합 안으로 넣는다
        
        $$
        B=\sum_{t=0}^T\mathbb E_{\tau\sim\pi_\theta} X_t
        $$
        
        - 전: trajectory마다 time step $t$에 걸쳐 $X_t$를 더하고, 샘플링한 trajectory들로 평균 낸다
        - 후: time step $t$마다 trajectory들에 걸쳐 $X_t$를 평균 내고, 그 평균들을 더한다
    - $X_t$는 $(s_t,a_t)$에만 의존하므로(trajectory의 나머지는 상관없다) 전체 trajectory로 평균 내는 것과 $(s_t,a_t)$로만 평균 내는 것이 같다
        - 그다음 $(s_t,a_t)$에 대한 joint expectation을 $s_t$에 대한 기댓값 안의 $a_t$ given $s_t$ 기댓값으로 쪼갠다
        
        $$
        B=\sum_{t=0}^T\mathbb E_{s_t,a_t} X_t=\sum_{t=0}^T\mathbb E_{s_t}\left[\mathbb E_{a_t\mid s_t} X_t\right]
        $$
        
    - 이제 $X_t$의 정의를 쓴다
        
        $$
        \begin{align*}
        \mathbb E_{a_t\mid s_t} X_t&=\mathbb E_{a_t\mid s_t}\nabla_\theta\log\pi_\theta (a_t\mid s_t)\,b(s_t)\\
        &= b(s_t)\,\mathbb E_{a_t\mid s_t}\nabla_\theta\log\pi_\theta (a_t\mid s_t)&\text{because }b\text{ doesn't depend on }a_t\text{!}\\
        &= b(s_t)\,\sum_{a_t}\pi_\theta(a_t\mid s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\\
        &= b(s_t)\,\sum_{a_t}\nabla_\theta\pi_\theta(a_t\mid s_t)&\text{the log derivative trick again}\\
        &=b(s_t)\nabla_\theta\sum_{a_t}\pi_\theta(a_t\mid s_t)\\
        &= b(s_t)\nabla_\theta 1&\text{sum of probabilities of all actions is 1}\\
        &= b(s_t)\cdot 0\\
        &= 0
        \end{align*}
        $$
        
        - $b$가 $s_t$에만 의존하기 때문에 $a_t\mid s_t$ 기댓값 밖으로 $b(s_t)$를 꺼낼 수 있다. 그래서 $\sum_a\pi(a\mid s_t)=0$이라는 사실을 쓸 수 있다
    - 다시 대입하면 다음을 얻는다
        
        $$
        B=\sum_{t=0}^T\mathbb E_{s_t}[0]=0
        $$
        
- baseline function으로 아주 흔한 선택은(PPO가 쓴다) $V_\psi(s_t)$다. 부분 sequence $s_t$가 주어졌을 때 *기대* reward를 추정한다
    
    $$
    \nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-V_\psi(s_t))\right]
    $$
    
    - $V_\psi$는 state에만 의존하므로 unbiased다
    - token마다 받는 신호: 생성의 이 시점에서 예상한 것보다 최종 reward가 높았는가, 낮았는가?
        - 나빠 보이던 response(작은 $V_\psi(s_t)$)를 좋은 response(높은 $R(\tau)$)로 바꾼 token이 credit을 더 받는다
    - advantage와의 연결
        - $R(\tau)$는 **Q function** $Q^\pi(s_t,a_t)$(이 action과 state에서 policy $\pi$를 따를 때의 기대 reward)의 샘플 하나다
        - $V_\psi(s_t)$는 **value function** $V^\pi(s_t)$(이 state만 주어졌을 때의 기대 reward, $V^\pi(s)=\sum_{a\sim\pi(\cdot\mid s)}Q^\pi(s,a)$)의 추정치다
        - 따라서 $R(\tau)-V_\psi(s_t)$는 **advantage function** $Q^\pi(s,a)-V^\pi(s)=A^\pi(s,a)$의 추정치다. 즉 state $s$에서 action $a$가 기대보다 얼마나 나은지를 나타낸다
            - 특히 샘플 하나 $\tau$의 reward $R(\tau)$로 $Q^\pi$를 추정하므로 advantage의 **Monte Carlo estimate**다
- baseline function은 대체로 모두 현재 state에서의 기대 return $\mathcal V^\pi(s_t)$를 추정하려 한다
    - learned value function [위 내용]
    - RLOO: prompt마다 response를 $G$개 샘플링하고, 그 prompt의 baseline은 *나머지* reward들의 평균이다(GRPO와 매우 비슷하다. $i$를 포함하는지와 std로 정규화하는지만 다르다)
    - batch estimate(REINFORCE++): batch의 평균 reward를 쓴다

**off-policy policy gradient**

- on-policy policy gradient의 문제는 gradient step마다 policy로 inference를 해야 한다는 점이다. policy는 아마 크게 바뀌지 않는데도 그렇다
- off-policy learning에서는 최적화하는 policy와 다른 policy로 rollout을 샘플링한다
    - 보통 이전 버전 policy $\pi_{\theta_\text{old}}$의 rollout으로 현재 policy $\pi_\theta$를 최적화한다
- **surrogate objective** $\mathcal J^\text{surrogate}$를 쓴다
    
    $$
    \mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_{t=1}^T\underbrace{\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}}_{r_t}\, R(\tau)\right]
    $$
    
    - 분수 $r_t=\pi_\theta/\pi_{\theta_\text{old}}$는 importance sampling 방식의 재가중 항이다
        - importance sampling 배경: $\mathbb E_{x\sim p} f(x)$를 원하는데 샘플이 $x\sim q$뿐이면 다음처럼 다시 쓸 수 있다
            
            $$
            \mathbb E_{x\sim p}\,f(x)=\mathbb E_{x\sim q}\,\left[\frac{p(x)}{q(x)}f(x)\right]
            $$
            
            - 직관적으로 $x$가 $q$보다 $p$에서 더 나올 법하면 그 샘플을 더 크게 센다. 반대도 마찬가지다
    - 원래 $\mathcal J(\theta)$와 다른 objective를 최적화한다는 점에 주의한다!
    - *진짜* off-policy 추정은 $\mathcal J(\theta)$를 importance sampling으로 다시 쓰기만 한다. 하지만 ratio의 곱 때문에 variance가 끔찍하게 크다
        
        $$
        \mathcal J(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\frac{P(\tau\mid\theta)}{P(\tau\mid\theta_\text{old})}\,R(\tau)\right]=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\prod_{t=1}^T\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,R(\tau)\right]
        $$
        
        - $\mathcal J^\text{surrogate}$는 사실상 이 곱을 time step별 항의 합으로 바꾼 것이다
- $\mathcal J^\text{surrogate}$에서 다음 **off-policy policy gradient**가 나온다
    
    $$
    \nabla_\theta \mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau)\right]
    $$
    
    - 이게 맞는 이유는 $r_t$에서 분자만 $\theta$에 의존한다는 점을 보고 log derivative trick을 쓰면 알 수 있다
    - 실제로는 다음으로 추정한다. $N$은 batch당 rollout 수다
    
    $$
    \hat g_\text{off-policy}=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\frac{\pi_\theta(a_t^{(i)}\mid s_t^{(i)})}{\pi_{\theta_\text{old}}(a_t^{(i)}\mid s_t^{(i)})}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})R(\tau^{(i)})
    $$
    

## PPO

- importance weight $r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$에 clipping 메커니즘을 도입했다
    - 일반 surrogate objective는 다음과 같다
        
        $$
        \mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_tr_tA_t\right]
        $$
        
        - PPO는 value function을 baseline으로 쓴다. 그래서 관례대로 policy gradient의 $R(\tau)$를 time step별 advantage $A_t=R(\tau)-V_\psi(s_t)$로 바꾼다
    - **clipped surrogate objective**는 $r_t$가 $[1-\epsilon,1+\epsilon]$ 안에 머물도록 clip한다
    
    $$
    \mathcal J^\text{CLIP}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\min(r_tA_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]
    $$
    
- clipping은 rollout batch 하나로 gradient step을 여러 번 밟을 때 안정성을 지켜준다
    - unbiasedness를 포기하는 대신 더 안정적인 update를 얻는다
    - $\pi_{\theta_\text{old}}$에서 너무 멀리 벗어나지 않는 한 clipped objective는 좋은 근사다
- clipping이 쓰이는 네 가지 경우
    - $r_t>1+\epsilon$이고 $A_t>0$이면 $(1+\epsilon)A_t$를 쓴다
        - $\nabla\mathcal J(\theta)$가 $\theta$에 의존하지 않는다 → gradient가 0이다
        - old policy보다 $a_t$의 확률을 이미 크게 올렸다. 좋은 action을 더 밀지 않는다
    - $r_t<1-\epsilon$이고 $A_t<0$이면 $(1-\epsilon)A_t$를 쓴다
        - gradient가 역시 0이다
        - 나쁜 action을 더 내리지 않는다
    - $r_t>1+\epsilon$이고 $A_t<0$이면 $r_tA_t$를 쓴다
        - $\nabla\mathcal J(\theta)$가 $\theta$에 의존한다 [off-policy policy gradient 참고]
        - 즉 $r_t$가 $>1+\epsilon$이 될 수 있다. $A<0$이라 나쁜 token이므로 내리도록 두는 것이 맞다
        - clipping은 비대칭이다. policy가 이미 advantage가 권하는 방향으로 움직일 때만 작동하고, 실수를 바로잡는 것은 막지 않는다
    - $r_t<1-\epsilon$이고 $A_t>0$이면 $r_tA_t$를 쓴다
        - $\nabla\mathcal J(\theta)$가 $\theta$에 의존한다 [off-policy policy gradient 참고]
- PPO는 현재 policy로 trajectory batch를 모은 뒤 clipped surrogate objective로 gradient step을 여러 번 밟는다

## RLHF

- $\pi_\theta$가 $\pi_\text{ref}$에서 너무 멀어지지 않도록 KL penalty를 넣는다
    
    $$
    \mathcal J^\text{RLHF}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[R(\tau)-\beta D_{\text{KL}}(\pi_\theta\mid\mid\pi_\text{ref})\right]
    $$
    
- 실제로는 KL penalty를 token마다 계산해 token별 reward에 합친다
- reward model 학습 방법
    - Bradley-Terry model에 따르면 $y_w$가 $y_l$보다 선호될 확률은 다음과 같다
        
        $$
        P(y_w\succ y_l)=\frac{\exp(R(x,y_w))}{\exp(R(x,y_w))+\exp(R(x,y_l))}=\sigma(R(x,y_w)-R(x,y_l))
        $$
        
    - classification head를 단 LM을 학습해 다음을 최대화한다
        
        $$
        \mathcal L(\varphi)=-\log P(y_w\succ y_l)
        $$
        
        - 이는 binary cross-entropy loss $-(y\log p+(1-y)\log(1-p))$에서 정답 label이 항상 $y=1$이고 $p=P(y_w\succ y_l)$(Bradley-Terrey 기준)인 경우와 같다

## GRPO

- GRPO는 reward $R(\tau)$를 샘플링한 trajectory 그룹 대비 $\tau$의 advantage로 바꾼다
    - 참고: $R(\tau)-b(s_t)$를 정규화 계수로 나누기도 하므로 깔끔하게 baseline 선택 하나로 보기는 어렵다
    - 질문 $q$에 대한 rollout 집합 $\{o^{(i)}\}_{i=1}^G$가 있고, 각 rollout은 token $o^{(i)}_1,…,o^{(i)}_T$로 이루어진다
        - 샘플링한 output마다 $R(q,o^{(i)})$로 reward $\mathbf r = \{r^{(i)}\}_{i=1}^G$를 계산한다
        - group-relative advantage 추정치 $A^{(i)}$는 다음과 같다
            
            $$
            A^{(i)}=\frac{r^{(i)}-\text{mean}(\mathbf r)}{\text{std}(\mathbf r)}
            $$
            
            - advantage $A^{(i)}$는 response 안의 모든 token에 똑같다
    - objective는 다음과 같다
        
        $$
        \mathcal J^\text{GRPO-CLIP}(\theta)=\frac 1G\sum_{i=1}^G\frac 1T\sum_{t=1}^T\min\left(r_tA^{(i)},\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A^{(i)}\right)
        $$
        
        - 여기서 $r_t$는 같은 token별 probability ratio다(여기서는 관례상 $a_t$ 대신 $o_t$를 쓴다)
            
            $$
            r_t=\frac{\pi_\theta(o_t\mid s_t)}{\pi_{\theta_\text{old}}(o_t\mid s_t)}
            $$
            
    - 샘플 그룹 대비 advantage를 계산해 critic(value function)이 필요 없게 만들어 PPO를 단순화한다
- GRPO는 세 가지 아이디어를 합친다
    - $\pi_{\theta_\text{old}}$를 쓰는 off-policy policy gradient: $\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_tr_tR(\tau)\right]$
    - clipping 메커니즘 [PPO]: $\mathcal J^\text{CLIP}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\min(r_t R(\tau),\text{clip}(r_t,1-\epsilon,1+\epsilon)R(\tau))\right]$
    - group normalization으로 advantage $A^{(i)}$ 계산 [DeepSeek R1]: $\mathcal J^\text{GRPO-CLIP}(\theta)=\frac 1G\sum_{i=1}^G\frac 1T\sum_{t=1}^T\min\left(r_tA^{(i)},\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A^{(i)}\right)$
- 알고리즘
    - policy model $\pi_\theta\leftarrow\pi_{\theta_\text{init}}$
    - 각 step마다(`n_grpo_steps`)
        - $\mathcal D$에서 batch $\mathcal D_b$를 샘플링한다
        - old policy model을 갱신한다 $\pi_{\theta_\text{old}}\leftarrow\pi_\theta$
        - 각 질문 $q\in\mathcal D_b$마다 output $G$개 $\{o^{(i)}\}_{i=1}^G\sim\pi_{\theta_\text{old}}(\cdot\mid q)$를 샘플링한다
        - 샘플링한 output마다 $R(q,o^{(i)})$로 reward $\{r^{(i)}\}_{i=1}^G$를 계산한다
        - group-relative advantage estimation으로 $A^{(i)}$를 계산한다
        - 각 train step마다(`n_train_steps_per_rollout_batch`)
            - GRPO objective를 최대화하도록 policy model $\pi_\theta$를 갱신한다
    - replay 메커니즘을 쓰는 continuous training으로 $r_\varphi$를 갱신한다
- GRPO는 사실상 학습한 baseline 대신 경험적 baseline을 계산한다
    - 기대 reward를 예측하도록 학습하지 않고, 샘플링해서 직접 잰다
    - compute는 더 쓰지만(prompt마다 completion $G$개) memory는 덜 쓴다(value network를 저장할 필요가 없다)
- Dr. GRPO는 GRPO의 두 문제를 고친다
    - GRPO에서는 $\text{std}(\mathbf r)$가 작으면(질문이 너무 쉽거나 너무 어려우면) reward가 증폭된다 → 그 그룹을 최적화하는 비중이 커진다
        - 너무 쉽거나 너무 어려운 문제에 가중치를 더 주는 bias다
    - GRPO에서는 reward를 rollout 길이 $\frac{1}{\lvert o_i\rvert}$로 정규화한다
        - 정답 response 중에서는 짧을수록 gradient가 크다 → 더 강하게 강화된다
        - 오답 response 중에서는 길수록 gradient가 작다 → 덜 벌받는다
            - 모델은 정답을 못 맞힐 것 같으면 아주 긴 답을 내면 된다고 배운다

## DPO

- KL-regularized objective $\mathcal J^\text{RLHF}$에는 closed-form 최적해가 있다
    
    $$
    \pi^*(y\mid x)=\frac{1}{Z(x)}\pi_\text{ref}(y\mid x)\exp\left(\frac1\beta R(y)\right)
    $$
    
    - 정리하면
        
        $$
        R(y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)}+\beta\log Z(x)
        $$
        
- DPO loss는 관측한 preference의 negative log likelihood일 뿐이다
    
    $$
    \mathcal L^\text{DPO}(\theta)=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]
    $$
    

# Precision

- **mixed precision** training
    - master weights는 FP32다
    - forward/backward pass용 weight는 BF16 copy를 쓴다
    - activation은 BF16으로 계산한다
    - gradient는 BF16으로 계산해 FP32로 accumulate한다
        - 큰 weight에 작은 gradient를 더하는 문제를 피한다
        - BF16은 0 근처 precision이 더 높아서 0.0001 같은 작은 gradient는 표현하지만, 갱신된 weight 1.0001은 표현하지 못한다
        - accumulate한 gradient만 FP32에 떨어지면 된다. 개별 gradient는 weight보다 아주 작은 경우가 많다
        - 각 parameter의 `.grad`는 parameter와 같은 dtype이므로 개별 gradient는 FP32로 cast된다
- 직관
    - matmul은 rounding noise에 강하므로 forward/backward는 BF16이어도 괜찮다
    - 개별 gradient가 아주 작아서 master weights는 FP32여야 한다
- activation은 weight보다 quantize하기 어렵다
- precision 선택지
    - FP32: full precision(4 bytes)
    - FP16: memory 절반(2 bytes)
    - BF16: memory는 FP16과 같지만 수치 안정성이 더 좋다(dynamic range가 더 넓다)
    - INT8: FP32의 4분의 1 memory, quantization이 필요하다(1 byte)
    - INT4: 더 작다
- data loading
    - `memmap`을 쓰면 데이터 전체를 한 번에 memory에 올리지 않는다
- BF16으로 model 불러오기
    - HF `.from_pretrained()`로 model을 불러올 때
        - `torch_dtype=torch.bfloat16`은 BF16이다. 권장한다
            - weight와 activation 모두 BF16이다
        - `load_in_8bit=True`는 quantization이다. `LLM.int8()`을 쓴다
            - weight는 INT8, activation은 FP16이다
            - weight 압축으로 memory를 아낀다
            - activation이 더 높은 precision이라 완전한 INT8 compute와는 다르다
    - `model.half()`나 `model.to(torch.bfloat16)`은 model을 FP16으로 바꾼다
        
        ```python
        model = MyModel()
        model.load_state_dict(torch.load('model.pt'))
        model = model.half()
        ```
        
    - 모든 weight가 BF16이고 연산도 BF16으로 돈다
    - 실제로 inference는 전부 BF16으로 돌려도 괜찮다
- `torch.autocast()`는 automatic mixed precision을 한다
    - weight는 FP32로 두고 연산마다 BF16이나 FP32를 골라 쓴다
        - 연산별 precision은 관리하지만 master weights는 관리하지 않는다
        - model 전체를 BF16으로 불러오는 것보다 memory를 더 쓰지만 더 안정적일 수 있다
    - matmul은 FP16(낮은 precision을 견딘다), softmax는 FP32(수치 안정성에 precision이 필요하다), layernorm은 FP32(reduction에 precision이 필요하다)
    - model이 FP32이고 쉽게 변환할 수 없을 때, 또는 순수 BF16에서 수치 문제가 보일 때 유용하다
    - `dtype`은 "낮은 precision" 타입을 지정한다
    
    ```python
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(x)
    ```
    
- `bitsandbytes`를 쓰려면 `nn.Linear`를 `bnb.nn.Linear8bitLt`나 `bnb.nn.Linear4bit`로 바꾼다
    
    ```python
    def replace_linear_with_8bit(model):
        """Replace all nn.Linear with bnb.nn.Linear8bitLt"""
        for name, child in model.named_children():
            if isinstance(child, nn.Linear):
                # Create quantized replacement
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
                # Copy weights (they'll be quantized when moved to CUDA)
                new_layer.weight = bnb.nn.Int8Params(
                    child.weight.data,
                    requires_grad=False,
                )
                if child.bias is not None:
                    new_layer.bias = nn.Parameter(child.bias.data)
                
                setattr(model, name, new_layer)
            else:
                # Recurse into child modules
                replace_linear_with_8bit(child)
        
        return model
    
    # Usage
    model = MyTransformer()
    model.load_state_dict(torch.load('model.pt'))
    model = replace_linear_with_8bit(model)
    model = model.to('cuda')  # quantization happens here
    ```
    

# Parallelism

- **data parallelism**은 batch를 device들에 나눈다. **model parallelism**은 forward pass 하나의 계산을 device들에 나눈다
    - FSDP는 memory 효율을 위해 parameter를 rank들에 shard한다. 하지만 각 rank는 필요하기 직전에 parameter를 all-gather해서 forward pass 전체를 여전히 계산한다
- data parallelism의 한계
    - $M<B$가 필요하다. $B$를 "critical batch size"보다 키우고 싶지 않으므로 꼭 좋은 조건은 아니다
    - model이 여전히 device 하나에 안 들어갈 수 있다(ZeRO stage 3도 device당 activation memory는 줄이지 못한다)
- **strong scaling**: 학습용 chip 수를 늘리면 throughput(FLOPs/second)도 비례해 늘어난다
- DP는 throughput을, TP/PP는 model memory를, SP는 activation memory를 scale한다
- 5D parallelism
    - **data parallelism**(DP): 데이터 batch를 device들에 나눈다
    - **tensor parallelism**(TP): model의 서로 다른 layer/stage를 device들에 나눈다
    - **pipeline parallelism**(TP): 개별 layer/weight matrix를 device들에 나눈다
    - **sequence parallelism**(SP): 입력 sequence 길이를 device들에 나눈다
    - **expert parallelism**(EP): MoE model의 서로 다른 expert를 device들에 나눈다

## Background: core collective operations

- **broadcast**(one to all, 같은 데이터): GPU 하나가 데이터를 갖고 다른 모든 GPU에 똑같은 copy를 보낸다
- **all-gather**(all to all): 각 GPU가 데이터 조각을 갖고, 모든 GPU가 전체 모음을 받는다
    - 한 축의 sharding을 *없앤다*
        
        $$
        \operatorname{AllGather}_Y:\mathbf A[I,J_Y]\rightarrow \mathbf A[I,J]
        $$
        
- **reduce-scatter**: 각 GPU가 reduce 전 데이터를 갖고, reduction으로 합친 결과를 GPU들에 shard한다
    - all-gather와 아주 비슷하지만 shard를 각각 두지 않고 더한다
    - 한 축에 sharding을 *더한다*
    
    $$
    \operatorname{ReduceScatter}_{Y,J}:\mathbf A[I,J]\{U_Y\}\rightarrow \mathbf A[I,J_Y]
    $$
    
- **all-reduce**: 모든 GPU가 reduce 전 데이터를 갖고, reduction으로 합쳐 모든 GPU가 최종 결과를 받는다
    
    $$
    \operatorname{AllReduce}_Y\mathbf A[I,J]\{U_Y\}\rightarrow A[I, J]
    $$
    

- ring all-reduce: reduce-scatter + all-gather(각 process는 이웃 둘과 통신한다) [animation]
    - 처음에는 각 GPU가 reduce 전 데이터를 갖는다
    - reduce-scatter: reduction으로 데이터를 합친다. 각 GPU는 reduce된 일부를 갖는다
        - 산술은 전부 하고, 중복 copy는 하지 않는다
    - all-gather: 모든 GPU가 reduce된 일부들의 전체 모음을 받는다
        - copy는 전부 하고, 산술은 하지 않는다
- all-gather, reduce-scatter, all-reduce의 통신 시간은 array 크기와 bandwidth에만 달려 있다. array를 shard하는 device 수에는 달려 있지 *않다*!
- reduce-scatter와 all-gather는 서로의 backward pass에 쓰인다
    - forward의 all-gather → backward의 reduce-scatter
        - all-gather는 같은 chunk를 모든 device에 broadcast하고, 각 device는 그걸로 서로 다른 downstream 계산을 한다
        - upstream gradient는 바깥으로 갈라지는 branch에서 더해진다($x=a+b$이면 $\partial f/\partial x=\partial f/\partial a \cdot\partial a/\partial x + \partial f/\partial a\cdot \partial a/\partial x$)
        - 이게 정확히 reduce-scatter다. 모든 upstream gradient를 원래 온 device로 더해 모은다
        - forward의 fan-out → backward의 합
    - forward의 reduce-scatter → backward의 all-gather
        - reduce-scatter는 많은 입력을 chunk 하나로 더한다
        - backward에서 덧셈 node는 upstream gradient를 각 피가수로 copy한다
        - 이게 정확히 all-gather다. 각 chunk의 gradient를 기여한 쪽으로 broadcast한다
        - forward의 합 → backward의 fan-out
- 따라서 all-reduce의 backward도 all-reduce다!

**partitioning notation**

- mesh의 축 이름은 $(X, Y)$이고 matrix $A$의 축은 $(I, J)$다
- $I_X$: $A$의 row를 device mesh의 column에 걸쳐 나눈다
    - $A$의 축 $I$(row)를 mesh 축 $X$를 따라(각 row를 따라) 나눈다
- $I_Y$: $A$의 row를 device mesh의 row에 걸쳐 나눈다
- $J_Y$: $A$의 column을 device mesh의 row에 걸쳐 나눈다
- $J_X$: $A$의 column을 device mesh의 column에 걸쳐 나눈다
- $I_{XY}$: $A$의 row를 펼친 $XY$ mesh의 모든 device에 걸쳐 나눈다
- $I$: $A$의 row를 나누지 않는다
- mesh 차원이 표기에 없으면 그 차원을 따라 데이터가 복제된다는 뜻이다
    - $Y$가 없으면 각 column이 같은 데이터를 갖는다

- matmul의 좋은 성질: 곱하는 matrix를 block으로 쓰면 곱도 block matmul로 쓸 수 있다
- matmul 네 가지 경우
    - case 1: 어느 matrix도 contracting dimension이 shard되지 않았다
        
        $$
        \mathbf A[I_X,J]\cdot \mathbf B[J, K_Y]\rightarrow \mathbf C[I_X, K_Y]
        $$
        
        - 통신이 필요 없다
        - 로컬 block matmul을 한다
        - 출력이 자연스럽게 원하는 방식으로 shard된다
    - case 2: $A$나 $B$ 중 하나의 contracting dimension이 shard됐다
        
        $$
        \mathbf A[I,J_X]\cdot \mathbf B[J, K]\rightarrow \mathbf C[I,K]
        $$
        
        - $A$의 shard를 all-gather해 모든 device가 전체 copy를 갖게 한 뒤 $B$와 곱한다
        
        $$
        \operatorname{AllGather}_X[I, J_X]\rightarrow \mathbf A[I, J]\\
        \mathbf A[I,J]\cdot\mathbf B[J, K]\rightarrow\mathbf C[I,K]
        $$
        
    - case 3: $A$와 $B$ 모두 contracting dimension이 shard됐다
        
        $$
        \mathbf A[I,J_X]\cdot \mathbf B[J_X, K]\rightarrow\mathbf C[I,K]
        $$
        
        - matmul은 가능하지만 각 device는 원하는 곱의 partial sum만 갖는다
            - $X$ 차원을 따라 device마다 다른 partial sum을 갖는다
            - $C[I,K]\{U_X\}$는 mesh 축 $X$를 따라 unreduced라는 뜻이다
        - $X$ 축에 걸쳐 all-reduce로 최종 합을 낸다
            
            $$
            \operatorname{AllReduce}_XC[I,K]\{U_X\}\rightarrow C[I,K]
            $$
            
            - 결과적으로 모든 device가 똑같이 다 더한 값을 갖는다
    - case 4: $A$와 $B$ 모두 non-contracting dimension이 같은 축을 따라 shard됐다

## Data parallelism

**naive data parallelism (DDP)**

- $B$ 크기 batch 안의 example을 $M$개 device에 나누고 gradient를 교환한다
- 단계
    1. 로컬 micro-batch로 forward pass를 돈다
    2. 모든 param의 grad를 계산한다
    3. GPU들에 걸쳐 gradient를 *all-reduce*해 각 GPU가 평균 G를 갖게 한다
    4. 각 GPU가 모든 param을 독립적으로 갱신한다
        - 이 단계에서 GPU마다 똑같은 일을 한다
- model이 device 하나에 들어가면 항상 이걸 써야 한다
- 통신은 backward pass에서만 일어난다
- device를 늘려 batch size를 얼마든지 키울 수 있다

**ZeRO stage 1: shard optimizer states**

- 각 device는 Adam vector의 $1/M$만 갖고, optimizer state를 가진 parameter만 갱신한다
- 단계
    1. 로컬 micro-batch로 forward pass를 돈다
    2. 모든 param의 grad를 계산한다
    3. gradient를 *reduce-scatter*한다. 각 GPU는 자기 parameter 부분집합의 reduce된 gradient를 갖는다
    4. 각 GPU는 자기 optimizer state 부분집합으로 자기 param 부분집합을 갱신한다
    5. 갱신된 parameter를 *all-gather*한다
- all-reduce 한 번의 비용은 reduce-scatter + all-gather와 같으므로 추가 통신 overhead가 없다. memory 이득이 공짜다

**ZeRO stage 2: shard optimizer states + gradients**

- gradient도 shard해서 각 device가 gradient의 $1/M$만 갖는다
- 단계
    1. 로컬 micro-batch로 forward pass를 돈다
    2. 각 layer의 gradient가 계산되는 즉시 *reduce-scatter*한다. 각 GPU는 자기 gradient 부분집합만 남기고 나머지는 해제한다
    3. 각 GPU는 자기 optimizer state 부분집합으로 자기 param 부분집합을 갱신한다
    4. 갱신된 parameter를 *all-gather*한다

**ZeRO stage 3: shard optimizer states + gradients + parameters (FSDP)**

- parameter도 shard해서 각 device가 parameter의 $1/N$만 갖는다
- 각 layer의 parameter를 필요한 시점에 *all-gather*해서 쓰고 버린다
- GPU 하나에 안 들어가는 model도 학습할 수 있다
- ZeRO stage 1, 2, 3은 통신 비용이 모두 같다

## **Pipeline parallelism**

- model parallelism을 순진하게 하면 layer마다 다른 GPU로 옮기고, layer 앞에서 hidden state를 해당 GPU로 옮긴다(이건 나쁘다. throughput이 늘지 않는다)
- 제대로 된 async pipeline parallelism
    
    ```python
    from torch.distributed.pipeline.sync import Pipe
    
    model = nn.Sequential(
        nn.Linear(512, 512).to('cuda:0'),
        nn.ReLU().to('cuda:0'),
        nn.Linear(512, 512).to('cuda:1'),
        nn.ReLU().to('cuda:1'),
    )
    
    model = Pipe(model, chunks=8)  # splits batch into 8 micro-batches
    output = model(x)
    ```
    

## **Tensor parallelism**

- FSDP와 tensor parallelism은 효과적으로 함께 쓸 수 있다
    - batch 차원 $B$를 shard하면 all-gather 크기가 줄고, FFN 차원 $F$를 shard하면 FSDP의 통신 overhead가 준다

- MLP(사이에 activation이 있는 linear layer 두 개)는 column parallel($W^\text{up}$) → activation → row parallel($W^\text{down}$)로 한다
    - 첫 weight matrix를 column 방향으로 나눈다
    - 각 device가 자기 shard를 계산하고 activation을 로컬에서 적용한다
    - 두 번째 weight matrix를 row 방향으로 나눈다
    - 각 device가 partial result를 계산한 뒤 *all-reduce*로 더한다
        - 핵심 통찰
            
            ```python
            y = h @ W2
              = [h_0, h_1] @ [W2_0]
                             [W2_1]
              = h_0 @ W2_0 + h_1 @ W2_1
              = y_0 + y_1
            ```
            

- 다른 sharding을 고르면 중간에 collective가 필요해진다
    - matmul $Y=XW$에서
        - $W$를 column sharding하면 각 device가 출력의 slice를 계산하므로 출력이 깔끔하게 shard된다
        - $W$를 row sharding하면 각 device가 전체 출력의 partial sum을 계산하므로 마무리하려면 reduction이 필요하다
- SwiGLU 방식 MLP에서는 $W_\text{gate}$와 $W_\text{gate}$를 모두 column-shard하고, element-wise 곱을 $F$ 기준으로 shard된 채 둔 뒤, $W_\text{down}$을 row-shard하고 마지막에 all-reduce한다
- 코드
    
    ```python
    class ColumnParallelLinear(nn.Module):
        def __init__(self, in_features, out_features, world_size, rank):
            super().__init__()
            self.out_features_per_rank = out_features // world_size
            self.rank = rank
    
            self.linear = nn.Linear(
                  in_features,
                  self.out_features_per_rank,
                  bias=False,
                  device=f"cuda:{rank}"
            )
        
        def forward(self, x):
            return self.linear(x)
    
    class RowParallelLinear(nn.Module):
        def __init__(self, in_features, out_features, world_size, rank):
            super().__init__()
            self.in_features_per_rank = in_features // world_size
            self.rank = rank
            
            self.linear = nn.Linear(
                    self.in_features_per_rank,
                  out_features,        
                  bias=False,
                  device=f"cuda:{rank}"
            )
        
        def forward(self, x):
            # x is partial: (batch, in_features_per_rank)
            # each GPU computes partial result
            partial = self.linear(x)
            
            # All-reduce to sum partial results across GPUs
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
            
            return partial
    
    class TensorParallelMLP(nn.Module):
        """
        column parallel -> activation -> row parallel requires only one all-reduce
        """
        def __init__(self, d_model, d_ff, world_size, rank):
            super().__init__()
            assert d_model % world_size == d_ff % world_size == 0
            self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
            self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        
        def forward(self, x):
            x = self.fc1(x)      # (batch, seq, d_ff // world_size)
            x = nn.functional.silu(x)
            x = self.fc2(x)      # (batch, seq, d_model), all-reduced
            return x
    ```
    
- attention에서는 각 device가 attention head 일부를 맡는다(head끼리 독립이므로)
    - MLP처럼 마지막에 all-reduce 한 번이면 된다
    - attention head는 서로 독립이라 head 기준으로 나누면 attention 계산 중에 통신이 필요 없다!
    - column parallel(QKV) → 계산 → row parallel(output)
    
    ```python
    class TensorParallelAttention(nn.Module):
        def __init__(self, d_model, num_heads, world_size, rank):
            super().__init__()
            assert num_heads % world_size == 0
            self.rank = rank
            
            # Each GPU handles a subset of heads
            # QKV projection for local heads only
            self.qkv = ColumnParallelLinear(
                d_model, 
                3 * d_model,
                world_size,
                rank
            )
            self.out_proj = RowParallelLinear(
                d_model,
                d_model,
                world_size,
                rank
            )
        
        def forward(self, x):
            batch, seq_len, _ = x.shape
            
            # Project to local Q, K, V
            qkv = self.qkv(x)
            qkv = qkv.reshape(batch, seq_len, 3, self.num_heads_per_rank, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            
            # Attention on local heads
            q = q.transpose(1, 2)  # (batch, local_heads, seq_len, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_out = nn.functional.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)
            
            # Output projection with all-reduce
            return self.out_proj(attn_out)
    ```
    

# Multimodality

- Vision Transformer(ViT): 이미지를 vector sequence로 바꾼 뒤 표준 transformer encoder를 돌린다
- LLaVA
    - model hidden dimension은 $D=4096$이고 CLIP hidden dimension은 1024다
    
    ```python
    Image [224, 224, 3]
        ↓ patchify
    Patches [256, 588]
        ↓ linear projection (independent)
    Patch embeddings [256, 1024]
        ↓ prepend CLS, add position embeddings
    Sequence [257, 1024]
        ↓ CLIP (transformer encoder with 24 layers)
    CLIP output [256, 1024]
        ↓ projection W applied independently per token
    Visual embeddings [256, 4096]
        ↓ concatenate with text embeddings [N, 4096]
    Full sequence [256 + N, 4096]
        ↓ LLM transformer
    ```
    

- LLaVA-NeXT dynamic resolution
    - 고해상도 이미지를 여러 crop으로 나눠 각각 encode한 뒤 이어 붙인다
- Qwen2-VL
    - 이전에는 입력 크기가 고정이었다(예: 224 x 224) → 항상 patch 256개
        - shape $[256, D]$의 absolute position embedding을 학습한다
    - 학습한 position embedding 개수가 고정이라는 이유만으로 고정 크기 이미지에 묶여 있었다
    - 이제는 2D-rope를 쓴다
        - patch 크기는 여전히 고정이다
        - position을 (row, column) 좌표로 encode한다
        - RoPE가 positional encoding을 그때그때 만든다
        - model이 임의의 해상도와 종횡비로 일반화할 수 있다
- 현재 패러다임
    - understanding: ViT encoder → feature
    - generation: pixel space에서 동작하는 diffusion
- CLIP
    - contrastive learning으로 학습한다
    - 올바른 짝의 text/image embedding 유사도를 최대화한다
