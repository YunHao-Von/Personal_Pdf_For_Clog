## 校正偏度损失  
依据贝叶斯公式:  
$$p_i(y|x) = \frac{p_i(x|y)p_i(y)}{\sum_{l=1}^C p_i(x|l)p_i(l)} \tag{1}$$  
上述贝叶斯公式有两点需要注意的：  
+ 通过相对概率可以很容易地计算出类先验概率。  
+ 类条件$\{p_i(x|y) | i \in [m] \}$在不同的客户端是相同的。  

受到上述观察的启发，本文提出了$p_i(y|x)$的另一种参数化方法。假设对于所有的$i \in [m]$，类条件$p_i(x|y)$可以参数化为$p_i(x|y) = \hat{q}\{x|y;\theta^*\}$，其中$\hat{q}\{x|y;\theta^*\}$可以是任意的条件概率函数。然后，$p_i(y|x)$就可以由$\theta^*$参数化为:   
$$p_i(y|x) = \hat{q}_i(y|x;\theta^*) = \frac{\hat{q}(x|y;θ^*)\pi^y}{\sum_{l=1}^C\hat{q}(x|l;\theta^*)\pi_i^l} , \\ 其中, \pi_i^y = \frac{n_i^y}{n_i} + δ, y \in [C]  \tag{2}$$    

其中，$\pi_i^y$近似于类先验概率$p_i(y)$，$n_i^y$是客户端$y$上的样本量，$\delta \gt 0$是为了数值稳定性而添加的一个小常数。在本地更新期间，客户端$i$使用其本地数据更新$θ_i$，这使得$θ_i$成为$θ^∗$的最大似然估计。  

下面的命题表明，当用上述重新参数化训练时，局部模型是同构的。  
<font color=red>假设客户端的标签分布是偏斜的。设$θ_i$为给定客户端$i$的局部数据的$θ^∗$的最大似然估计。那么$s^2$几乎肯定收敛于零。</font>  

我们定义$\hat{q}(x|y;\theta^*) = exp(f^y_{\theta^*}(x))$，接着，我们最大化 $\hat{q}(x|y;\theta^*)$,这相当于最大化以下目标函数：  
$$\min_{θ_i} \frac{1}{n_i} \sum_{j=1}^{n_i}l_{cce} (f_{θ_i}(x_{ij}'),y_{ij},\pi_i) \tag{3}$$

$l_{cce}$定义为:  
$$l_{cce} (f_{θ_i}(x_{ij}),y_{ij},\pi_i) = -log \sigma^{y_{ij}}(f_{θ_i}(x_{ij}') + log\pi_i) \tag{4}$$

最小化上述CCE损失可以减轻局部模型的异构性，从而提高全局模型的收敛性和性能。在以前的FAT方法中，异构局部模型倾向于给多数类较高的分数，而给少数类较低的分数。相比之下，我们的CalFAT鼓励局部模型通过在logits中添加分类先验对数$\pi_i$来给少数类给出更高的分数。与MixFAT在自然和对抗数据上训练局部模型不同的是，我们的CalFAT只在对抗样本上训练局部模型。  

受[40]的启发，我们通过最大化以下校准的Kullback-Leibler (CKL)散度损失来生成对抗性示例:  
$$x_{ij}' = argmax_B l_{ckl} (f_{θ_i}(x'_{ij}), f_{\theta_i}(x_{ij}),\pi_i) \tag{5}$$   

CKL损失函数为：  
$$l_{ckl} (f_{θ_i}(x'_{ij}), f_{\theta_i}(x_{ij}),\pi_i) = \sum_{y=1}^C σ^y(f_{θ_i}(x_{ij}) + log \pi_i) log σ^y (f_{θ_i}(x_{ij}') + log \pi_i) \tag{6}$$  

在按照上述过程训练局部模型的特定时期后，每个客户端i将模型参数$θ_i$上传到服务器进行聚合。为了与最新的FAT方法[47,9]保持一致，我们采用了最广泛使用的fedag[23]作为默认聚合框架。我们的方法与其他FL框架(例如，FedProx[16]和Scaffold[11])兼容，我们将在4.1节中展示。

-----
### CalFAT算法:  
输入:  
+ 客户端$i$  
+ 全局模型参数$\hat{\theta}$。  
+ 本地数据集$D_i$。  
+ 本地epoch数$E$。  
+ 正的常数$δ$。  

算法：  
+ 在客户端$i$利用数据集$D_i$计算$\pi_i$，$\pi_i^y = \frac{n_i^y}{n_i} + \delta, y\in [C]$。  
+ for local_epoch = 1...E:  
  + for j in range(1,n_i):  
    + 从数据集$D_i$中采样$(x_{ij},y_{ij})$  
    + 生成对抗样本$x' = argmax_B l_{ckl}(f_{θ_i}(x'_{ij}),f_{θ_i}(x_{ij}),\pi_i)$  
  + $\theta_i = \theta_i - \eta \frac{1}{n_i} \sum_{j=1} ^{n_i} ∇ l_{cce}(f_{θ_i}(x'_{ij}),y_{ij},\pi_i)$
+ 返回 $θ_i$


-----

