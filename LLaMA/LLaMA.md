Paper: https://arxiv.org/abs/2302.13971
# **1. Introduction**
Some LLMs trained on massive corpora can achieve the few-shot properties (perform new tasks from textual instructions or from new examples) whenever the model is scaled to a sufficient size. Some research conclude on an assumption that more parameters will lead to better performance. However, this disregard the strategies of inference budget.
This work is to train a series of language models that achieve the best possible performance at various inference budgets, by training on more tokens than what is typically used. Some remarkable results are:
- The resulting models ranges from 7B to 65B parameters with competitive performance compared to the best existing LLMs.
- The authors only used publicly available data making the work compatible with open-sourcing.
# **2. Approach**
## **2.1. Pretraining data**
The author used a combination of data from multiple sources which is summarized in the below table. Overall, the entire training dataset contains roughly 1.4T tokens after tokenization.
![[Pasted image 20241230141546.png | 400]]
## **2.2. Architecture**
The author leverage many improvements that were subsequently proposed.
- **Pre-normalization**: the author apply input normalization RMSNorm of each transformer sub-layer instead normalizing the output for training stability.
- **SwiGLU activation function**: the author replaced ReLU non-linearity by SwiGLU activation function with dimension of $\frac{2}{3}4d$ to improve the performance.
- Rotary Embeddings: the author removed absolute positional embeddings and 

