I personally found it hard to grasp back-propagation all at once when I first approached it. In this exposition, we offer a matrix calculus based derivation that is compact and easy to understand. Tensor calculus is amazing for its generality and power and matrix calculus has some shortcomings. For instance, the derivative of a matrix-valued function defined over matrices cannot be represented by a matrix. Despite its limitation, I believe that matrix calculus can be insightful here as we fortunately manage to maintain all information structure in matrices. 

# Necessary Derivatives

Throughout, we will follow [Numerator-Layout](https://en.wikipedia.org/wiki/Matrix_calculus) (which is better practice for many reasons!). We are going to need to a few simple lemmas before we jump into the main material. 

**Lemma 1.**

Let $x \in \mathbb{R}^n$ and $W \in \mathbb{R}^{m\times n}$:

$$
\frac{d}{dx}(Wx) = W
$$

*Proof.*

$$
\frac{d}{dx}(Wx)_{ij} = \frac{\partial }{\partial x_j}(\sum_{k=1}^{m} W_{ik}x_k) = W_{ij}
$$

**Lemma 2.** 

Let $f: \mathbb{R}^m \to 1$ and $W \in \mathbb{R}^{m \times n}$:

$$
\frac{d}{dW} f(Wx) = x \, \nabla f(Wx)^T
$$

*Proof.*

$$
    y:= Wx \implies \frac{\partial (Wx)}{\partial W_{ji}} = \overset{\overset{j\text{th element}}{\downarrow\hspace{0.7cm}}}{\begin{pmatrix}0 &\dots& 0& x_i& 0& \dots& 0\end{pmatrix}_{1\times m}^T} = x_i e_j
$$

$$
\begin{align*}
&\frac{d}{dW_{ji}} f(Wx) = \frac{\partial f}{\partial (Wx)} \frac{\partial (Wx)}{\partial W_{ji}} = \underbrace{\nabla f(Wx)}_{1 \times m}~\underbrace{x_i e_j }_{m\times 1}~ \quad \text{by chain rule}\\
&\[\frac{d}{dW} f(Wx)\]_{ij} = \frac{\partial}{\partial W_{ji}} f(Wx) = x_i\,[\nabla f(Wx)]_j
\end{align*}
$$

# Modeling

## Circuit

We represent the output of layer $k$ with $y^{(k)}$ which is a vector of arbitrary size $l^{(k)}$. The activation functions $\sigma_{k+1}: \mathbb{R}^{l^{(k+1)}} \longrightarrow \mathbb{R}^{l^{(k+1)}}$ are assumed to be differentiable which means we are not covering relu here (the fix might be covered in a later blog!).  We have $L$ layers in total with the output of last layer implementing the function we wish to approximate. To chain the layers together, we must use a recursive relationship:

$$
y^{(k+1)} = \sigma_{k+1}(W^{(k+1)}y^{(k)} + b^{(k+1)})
$$

with base case:

$$
y^{(1)} = \sigma_1(W^{(1)}x + b^{(1)})
$$

## Cost Function

Let us stick to the basic least-squared cost for our function approximation task. For the purposes of optimization we need a cost function that depends on all the weight matrices and biases:

$$
\mathbf{C}: \mathbb{R}^{l^{(0)}\times l^{(1)}} \times \mathbb{R}^{l^{(1)}} ... \times \mathbb{R}^{l^{(L-1)}\times l^{(L)}} \times \mathbb{R}^{l^{(L)}} \longrightarrow \mathbb{R}_{>0}\\
W^{(1)}, b^{(1)}, \dots, W^{(L)}, b^{(L)} \longmapsto C(W^{(1)}, b^{(1)}, \dots, W^{(L)}, b^{(L)})
$$

### Confusing Notation Alert

For brevity of notation, we denote a large number of different functions by the same letter $C$. What function we are referring to depends on the number and ***type*** of its inputs. 

What is to be noticed is that they are all equal to each other given “the right” inputs and that’s why they get the same name. 

Here:

$$
C(y^{(L)}) = C(W^{(L)}, b^{(L)}, y^{(L-1)}) = C(W^{(L)}, b^{(L)}, W^{(L-1)}, b^{(L-1)}, y^{(L-2)}) = \dots
$$

If you already know $y^{(L)}$, then that is all you need to know in order to compute the cost. Similarly you can argue that $W^{(L)}, b^{(L)}, y^{(L-1)}$ is enough information to compute the error. 

Suppose we are given training data $\mathcal{D} = \{(s_i, t_i)\,,\, i \in [d]\}$:

$$
C(y^{(L)}) = \frac{1}{2d} \sum_{i=1}^{d}{||{t - y^{(L)}(s_i)}||}^2
$$

And again, the function $y^{(L)}$ above is different from the activation function $\sigma_{L}$ that also gives out the output of the last layer $L$. In summary, lots of functions that output the same thing!

# Derivation

## Forward Pass

To give feedback to the system regarding the prediction error we compute the derivative (which is the transpose of the gradient in our setting):

$$
\frac{\partial C}{\partial y^{(L)}} = \frac{1}{d}\sum_{i=1}^{d}(y^{(L)}(s_i) - t_i)^T \quad \text{base case: forward pass}
$$

## Backward Pass

The nice thing about matrix calculus is that it takes care of everything under the hood of chain rule for us and we only need to mindlessly get the ordering right, for example:

$$
\frac {\partial y^{(k+1)}}{\partial y^{(k)}} = \frac{\partial y^{(k+1)}}{\partial (W^{k+1}y^{(k)}+b^{(k)})} \frac{{\partial (W^{k+1}y^{(k)}+b^{(k)}})}{\partial y^{(k)}} = \nabla \sigma_{k+1} W^{(k+1)}
$$

Now following chain rule we write:

$$ \begin{align*} &\frac{d}{dW_{ji}} f(Wx) = \frac{\partial f}{\partial (Wx)} \frac{\partial (Wx)}{\partial W_{ji}} = \underbrace{\nabla f(Wx)}_{1 \times m}~\underbrace{x_i e_j }_{m\times 1}~ \quad \text{by chain rule}\\ &[\frac{d}{dW} f(Wx)]_{ij} = \frac{\partial}{\partial W_{ji}} f(Wx) = x_i\,[\nabla f(Wx)]_j \end{align*} $$

where the correct explicit way to read the first equation is:

$$
\frac{\partial C(W^{(L)}, b^{(L)}, \dots, W^{(k+2)}, b^{(k+2)}, y^{(k+1)}(y^{(k)}))}{\partial y^{(k)}} = \dots
$$

$$
\begin{align}    \frac{\partial C}{\partial y^{(k)}} &= \frac{\partial C}{\partial y^{(k+1)}}\,\nabla \sigma_{k+1}\,W^{(k+1)} \\    \frac{\partial C}{\partial W^{(k)}} &= y^{(k-1)}\frac{\partial C}{\partial y^{(k)}}\,\nabla \sigma_k\\ \frac{\partial C}{\partial b^{(k)}} &= \frac{\partial C}{\partial y^{(k)}}\, \nabla \sigma_k\end{align}
$$

where $\nabla \sigma_k$ is a diagonal matrix and that is why most places choose to write out backprop with Hadamard product. Recall that if $x, y \in \mathbb{R}^n$, then $x \odot y := \text{diag}(x)\, y$.

# Simulation

The code for forward pass is embarrassingly simple:

```python
for k in range(1, self.depth):
		self.y[k] = self.sigma(self.W[k] @ self.y[k-1] + self.b[k])
		self.nabla_sigma[k] = np.diagflat(self.sigma_derivative(self.y[k]))
```

For backward pass the three main equations lead to only three lines of code!

```python
for k in range(1, self.depth):
        
		# clip
    self.dC_dy[-k] = self.clip(self.dC_dy[-k])
        
		# first equation
		self.dC_dy[-1-k] = self.dC_dy[-k] @ self.nabla_sigma[-k] @ self.W[-k]

		# second equation
    self.W[-k] += - self.lr * (self.y[-1-k] @ (self.dC_dy[-k] @ self.nabla_sigma[-k])).T
        
		# third equation
    self.b[-k] += - self.lr * (self.dC_dy[-k] @ self.nabla_sigma[-k]).T
```

I wrote the code with pedagogical concerns in mind and not for optimality at all. For example, we are wastefully multiplying a diagonal matrix with a vector when we could’ve used Hadamard product. But I didn’t want to make the code longer by using np.multiply. The obsession here is making the code as short and as close to the equations as possible. 

Here is an illustration of how the back-propagation approximates a sinusoidal function:

![Unknown.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/93aadc3a-edf1-4ac9-ae85-618a264dbe15/Unknown.png)

Python can be found here: https://colab.research.google.com/drive/1ZWUUhFB1D2dhJfpEUHtaczFnwVnmciH-?usp=sharing
