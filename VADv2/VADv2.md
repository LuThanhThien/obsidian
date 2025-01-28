Paper: [[2402.13243] VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning](https://arxiv.org/abs/2402.13243)
Source: https://github.com/hustvl/VAD
# **1. Introduction**
Learning a human-like driving policy is promising but witnesses many challenge due to uncertainty and non-deterministic nature of planning. From perspective of statistics, the action (including time and speed) is highly stochastic, affected by many latent factors that cannot be modeled.
In this work, the author proposed VADv2 probabilistic planning to cope with the uncertainty of planning.
The planning action space is a high-dimensional continuous spatiotemporal space. The author applied a probabilistic field function to map action space to the mass probabilistic distribution using discrete large planning vocabulary since continuous space is not feasible.
Probabilistic modeling compare with deterministic modeling has some advantages:
- Effectively captures the uncertainty in planning and achieve more accurate and safe planning performance.
- Models the correlation between each action and environment.
- Flexible in inference stage since it multi-mode planning results can be easily combined with  rule-based and optimization  -based planning methods.

The research contributions can be listed as bellow
- A probabilistic field to map from the action space to the probabilistic distribution, and learn the distribution of action from largescale driving demonstrations.
- VADv2: e2e driving model which transforms sensor data into environmental token embeddings, outputs the probabilistic distribution of action, and samples one action to control the vehicle.
![[VADv2 outline|1200]]
# **2. Related works**
## **Perception**
Perception is the foundation of autonomous driving, where a **unified representation like Bird’s Eye View (BEV)** aids in integrating downstream tasks. Key perception-related works include:
- **LSS**: Transforms perspective views to BEV by explicitly predicting depth for image pixels.
- **BEVFormer**: Avoids explicit depth prediction using spatial and temporal attention for enhanced detection.
- **Subsequent Works (e.g., BEVerse, ST-P3)**: Improve temporal modeling and BEV transformation for downstream tasks.
- **HDMapNet**: Converts lane segmentation into vector maps via post-processing.
- **VectorMapNet**: Predicts vector map elements in an autoregressive manner.
- **MapTR**: Introduces permutation equivalence and hierarchical matching for improved mapping.
- **LaneGAP**: Employs path-wise modeling for generating lane graphs effectively.
## **Motion prediction**
Motion prediction **forecasts future trajectories of traffic participants** to assist ego vehicle planning. Key methods include:
- **End-to-End Methods**: Combine perception and motion prediction for joint optimization.
- **Rasterized Representations**: Use CNNs for scene representation and trajectory prediction.
- **Vectorized Representations**: Employ Graph Neural Networks or Transformers for feature extraction.
- **Occupancy and Flow-based Prediction**: Predict dense motion instead of agent-level waypoints.
- **Gaussian Mixture Models (GMM)**: Regress multi-modal trajectories with uncertainty modeling.
## **Planning**
Planning focuses on generating control decisions for autonomous vehicles. Key research directions include:
- **Black-Box Models**: Use sensor data directly for control signal prediction.
- **Reinforcement Learning**: Explore driving behaviors in closed-loop simulations to achieve human-level performance.
- **Imitation Learning**: Learn expert driving behavior to mimic human-like planning.
- **End-to-End Models**: Integrate perception, motion prediction, and planning into a single data-driven model.
    - **UniAD**: Integrates perception and prediction to enhance planning.
    - **VAD**: Uses vectorized scene representations for planning without dense maps.
## **Large Language Model in Autonomous Driving**
Large Language Models (LLMs) bring interpretability and logical reasoning to autonomous driving, enabling advancements in scene understanding, evaluation, and planning. Key works in this domain include:
- **QA-based Scene Understanding**: LLMs are used for driving scene understanding and evaluation through question-answering tasks.
- **DriveGPT4**: Combines historical video, text, and control signals to predict answers and driving control signals using LLMs.
- **LanguageMPC**: Leverages Chain of Thought reasoning on language-encoded perception results and HD maps to predict planning actions.
- **VADv2**: Inspired by GPT, models planning as a stochastic process. It discretizes the action space into a planning vocabulary and samples actions from a learned probabilistic distribution based on driving demonstrations.
# 3. Method
## 3.1. Scene Encoder
![[VADv2 pipeline.png| 700]]
The author used an encoder to convert multi-view image sequence into scene tokens $E_{env}$ including **4 different types of tokens**: **agent tokens** (information of agent's speed, location, orientation, etc.), **map tokens** (lanes, road boundary, centerline, etc.), **traffic element tokens** (states of traffic elements: traffic light signals and stop signs), **image tokens**. Auxiliary information like navigation and ego state are also encoded $\{E_{navi},E_{state}\}$ using $\text{MLP}$.

## 3.2. Probabilistic Planning
The author models the planning policy as an **environment-conditioned nonstationary stochastic process** formulated as $p(a|o)$. Planning action space is approximated as a spatiotemporal space $\mathbb{A} = \{a|a\in \mathbb{R}^{2T}\}$ and is sampled one at each time step to control the vehicle. 
Planning actions space is **discretized to a large planning vocabulary** $\mathbb{V}=\{a^i\}^N$ due to the infeasibility of directly fitting continuous planning action space. When the trajectory is converted into control signals, the control signal values need to satisfy the kinematic constraints of the ego vehicle.
Each action in the planning vocabulary is a waypoint sequence $a=(x_1,y_1,x_2,y_2,...,x_T,y_T)$, each waypoint corresponds to a future timestamp. The continuous radiance field is modeled over the 5D space $(x, y, z, \theta, \phi)$. After discretization, each **action (trajectory) is encoded into a high-dimensional planning token embedding** $E(a)$ using a cascaded Transformer decoder.
$$E(a)=\text{Cat}[\Gamma(x_1),\Gamma(y_1),\Gamma(x_2),\Gamma(y_2),...,\Gamma(x_T),\Gamma(y_T)]$$
$$\Gamma(\text{pos})=\text{Cat}[\gamma(\text{pos},0),\gamma(\text{pos},1),...,\gamma(\text{pos},L-1)]$$
$$\gamma(\text{pos},j)=\text{Cat}[\text{cos}(\text{pos}/10000^{2\pi j/L}),\text{sin}(\text{pos}/10000^{2\pi j/L})]$$
Where $\Gamma$ is an encoding function that maps each coordinate from $\mathbb{R}$ ($\text{pos}$) into a high dimensional embedding space $\mathbb{R}^{2L}$. Finally to achieve the probability, we feed these token embeddings into a Transformer with $E(a)$ as query and $E_{env}$ as key and value. Adding the result with auxiliary embeddings ($E_{navi}$ and $E_{state}$) and pass through $\text{MLP}$ layer.
$$p(a)=\text{MLP}(\text{Transformer}(E(a),E_{env})+E_{navi}+E_{state}),$$
## **3.3. Training**
Three types of supervision are taken into account for training including: distribution, conflict and scene token loss
$$\mathcal{L}=\mathcal{L}_{distribution}+\mathcal{L}_{conflict}+\mathcal{L}_{token}$$
- **Distribution loss:** the probabilistic distribution is learned from large-scale driving demonstrations using KL divergence. Trajectories close to the ground truth are less penalized with smaller loss weights. 
- **Conflict loss:** If one action in action vocabulary is conflict with other agents' future motion or road boundary, it is considered as a negative sample
- **Scene Token loss:** Map tokens (focal loss for map classification), agent tokens (detection loss and motion prediction loss) and traffic element tokens (focal loss with traffic light and stop sign as classification task) are supervised with corresponding supervision signals.
## **3.4. Inference**
Goal action is sampled from the distribution with **highest probability at each timestep**, and **converted into control signals** (steer, throttle, and brake) using **PID controller**.
In real-world applications, a good practice is to to **sample top-K actions** and post process with **rule-based wrapper and optimization-based post-solver**. Action probability can be considered as a confident score to act as the judgement condition for switching between conventional and learning-based **PnC** (Planning and Control).
