![[UniAD demo.png]]
Github: https://github.com/OpenDriveLab/UniAD
Conference: https://www.youtube.com/watch?v=cyrxJJ_nnaQ
# **1. Introduction**
## **1.1. Related Works**
Autonomous driving algorithms are assembled with a series of tasks, including **detection, tracking, mapping in perception; and motion and occupancy forecast in prediction**
![[Autonomous driving various designs.png]]
- **Standalone models:** design simplifies the R&D difficulty across teams, it bares the risk of information loss across modules
- **Multi-task Framework (MTF):** plugging several task-specific heads into a shared feature extractor (models such as *Transfuser*, *BEVerse*). It can leverage feature abstraction, extendable but causing undesirable “negative transfer”.
- **End-to-end autonomous driving:** The choice and priority of preceding tasks should be determined in favor of planning. The system should be **planning-oriented**.
Desirable system should be **planning-oriented** as well as properly **organize preceding tasks to facilitate planning**. Intuitive resolution would be to perceive surrounding objects, predict future behaviors and plan a safe maneuver explicitly. However, some previous works more or less fail to consider certain components.
## **1.2. UniAD**

**Unified Autonomous Driving** is an algorithm framework to leverage five essential tasks toward a safe and robust system as depicted. A key component is the query-based design to connect all nodes.
**Query-based design** compared with classic **bounding box representation**:
- In traditional object detection tasks, objects are represented using bounding boxes which **rely heavily on fixed, local features**. Predictions based on bounding boxes **accumulate errors** as they propagate through the system.
- Query-based design allow the model to “see” **larger receptive field** of input data, take into account more global context. Also, queries can **adapt dynamically** based on the input data, and are not restricted to encoding simple representations (like bounding boxes), thus it can **represent relations between agents**.
# **2. Methodology**
## **2.1. Overview**
![[UniAD Pipeline | 1200]]
UniAD comprises four transformer decoder-based perception and prediction modules and one planner in the end. In detail, UniAD is composed of:
1. **Feature extractor and encoder:** a sequence of multi-camera images is fed into the **feature extractor**, and the resulting perspective-view features are transformed into a unified **bird’s-eye-view (BEV) feature B** by **BEV encoder** in BEVFormer (this block is not confined and can be utilized as other types)
2. **TrackFormer:** learnable embeddings (track queries) inquire about the agents’ information form B to **detect and track agents**
3. **MapFormer:** takes map queries and performs panoptic segmentation of the map.
4. **MotionFormer:** captures interactions among agents, maps and forecast per-agent future trajectories in a joint prediction way.
5. **OccFormer:** employs the BEV feature B as queries, equipped with agent-wise knowledge as keys and values to predict multi-step future occupancy with agent identity preserved.
6. **Planner:** predict plan result such that avoids collisions based on occupied regions predicted by **OccFormer**.
## **2.2. Perception: Tracking and Mapping**
### **TrackFormer**
End-to-end detection and multi-object tracking (MOT) **without non-differentiable post-association**. Former contains N layers and the final output state **dynamic agent queries $Q_A$** provides knowledge of $N_a$ valid agents for downstream prediction tasks.

![[TrackFormer | 1200]]
- **Detection queries:** initialized detection queries are responsible for detecting new born agents.
- **Track queries:** keep modeling these agents detected in previous frame.
- **Ego-vehicle query:** the author introduced one particular **ego-vehicle query in the query set** to explicitly model the self-driving vehicle itself, which is further used in planning.
### **MapFormer**
The author design MapFormer based on 2D panoptic segmentation method Panoptic SegFormer (https://arxiv.org/abs/2109.03814), **each query represents a map element**. MapFormer also has N stacked layers whose output results of each layer are all supervised, only the updated **static map queries $Q_M$** in the last layer are forwarded to MotionFormer for agent-map interactions.

![[MapFormer| 1200]]
### **Training and inference stages**

| **Aspect**               | **Training Stage**                                                               | **Inference Stage**                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Query Initialization** | All queries are considered **detection queries** in the first frame.             | **Track queries** can persist across frames for a longer horizon.                                                                |
| **Query Matching**       | - Detection queries matched with ground truth using the **Hungarian algorithm**. | Queries updated based on **classification scores** instead of ground truth.                                                      |
|                          | - Queries updated via the **Query Interaction Module (QIM)**.                    |                                                                                                                                  |
|                          | - Track queries matched directly by **track ID**.                                |                                                                                                                                  |
|                          | - Detection queries matched to remaining **newborn objects**.                    |                                                                                                                                  |
| **Matching Metric**      | Uses **3D IoU** to filter matched queries (threshold > 0.5).                     | Uses **classification scores**:<br> - > 0.4 for detection queries.<br> - > 0.35 for track queries.                               |
| **Query Updates**        | Only predictions with **3D IoU > 0.5** are stored and updated.                   | - Detection queries are filtered with scores > 0.4.<br> - Track queries are filtered with scores > 0.35.                         |
| **Handling Occlusion**   | Not explicitly mentioned.                                                        | Uses a **lifecycle mechanism**:<br> - Track queries are **removed only if classification score < 0.35** for **2s** continuously. |
| **Sequence Processing**  | Each sequence starts fresh with all detection queries.                           | Each frame is processed **sequentially**, with persistent track queries.                                                         |
## **2.3. Prediction: Motion Forecasting**
MotionFormer predicts all agents’ multimodal future movements, i.e., **top-k possible trajectories**, in a scene-centric manner. Formally the output formation is formulated as
$$\{\hat{x}_{i,k} \in \mathbb{R}^{T \times 2} \mid i = 1, \dots, N_a; \, k = 1, \dots, \mathcal{K}\}$$
Where $i$ **indexes the agent**, $k$ indexes the **modality of trajectories** and $T$ is the **length of prediction horizon**. 
### **MotionFormer**
Composed of N layers, and each layer capture 3 types of interaction: **agent-agent, agent-map and agent-goal point**. 
- **Agent-agent and agent-map interactions:** For each motion query $Q_{i,k}$ , its interactions between **other agents $Q_A$ or map elements $Q_M$** are built with **standard transformer decoder layers** and could be formulated as:
$$Q_{a/m}=\text{MHCA}(\text{MHSA}(Q), Q_A/Q_M)$$
	where $\text{MHCA}$ and $\text{MHSA}$ denote multi-head cross-attention and multi-head self-attention respectively. 
- **Agent-goal point** attention is devised via deformable attention as follows:
$$Q_g=\text{DeformAttn}(Q, \hat{x}_T^{l-1},B)$$
	where $\hat{x}^{l-1}_T$ is the **endpoint of the predicted trajectory of previous layer**. **Deformable Attention** module performs sparse attention on spatial feature around the reference point, thus the **predicted trajectory is further refined as "aware" of the endpoint surrounding**. 
These all three interactions are ==**modeled in parallel**== then they are concatenated and passed to a **multi-head perceptron (MLP)**, resulting a query context $Q_{cxt}$. Subsequently, each agent's trajectories are constructed applied **Gaussian Mixture Model**, where $\hat{x} \in \mathcal{R}^{\mathcal{K}\times\mathcal{T}\times5}$.  
![[MotionFormer | 1200]]
### Motion queries
**Input queries** of each MotionFormer layer is composed of **2 components**:
- **Query Context $Q_{ctx}$** : produce by preceding layer as above
$$Q_{cxt}=\text{MLP}(\text{Concat}(Q_a,Q_m,Q_g))$$
- **Query Position $Q_{pos}$ :** integrates the positional knowledge in four-fold and calculate using **nusoidal positional embedding follow by MLP block**
	- (1) the position of **agent-level anchors** $I^{a} \in \mathbb{R}^{T \times 2}$
	- (2) the position of **scene-level anchors** $I^{s} \in \mathbb{R}^{T \times 2}$. This can be obtained from agent-level anchor being rotated and translated into global coordinate frame.
$$I_{T}^s=I_{i,T}^s=R_iI_T^a+T_i$$
	- (3) **current location** of agent $i$, $\hat{x}_0$
	- (4) **predicted goal point** $\hat{x}_T^{t-1}$, initially set as scene-level anchors $\hat{x}_{T}^0=I^s\in \mathbb{R}^{T \times 2}$
$$Q_{pos}=\text{MLP}(\text{PE}(I^s))+\text{MLP}(\text{PE}(I^a))+\text{MLP}(\text{PE}(\hat{x}_0))+\text{MLP}(\text{PE}(I^s))$$
Query context and query position are all built up a motion query with shape $\in \mathbb{R}^{\mathcal{K} \times \mathcal{D}}$. 

![[Agent-goal interaction module.png | 500]]
### **Non-linear optimization**
The author adopt a non-linear smoother to adjust the target trajectories and make them physically feasible given an imprecise starting point predicted by the upstream module. This is applied during training so it does not effect inference performance.
$$
\tilde{\mathbf{x}}^* = \arg\min_{\mathbf{x}} c(\mathbf{x}, \tilde{\mathbf{x}})
$$
Where $\tilde{\mathbf{x}}$ is the ground truth and $\tilde{\mathbf{x}}^*$ is smoothed trajectory, while $x$ is generated by multiple-shooting. The cost function is as follows:
$$
c(\mathbf{x}, \tilde{\mathbf{x}}) = \lambda_{xy} \|\mathbf{x}, \tilde{\mathbf{x}}\|_2 + \lambda_{\text{goal}} \|\mathbf{x}_T, \tilde{\mathbf{x}}_T\|_2 + \sum_{\phi \in \Phi} \phi(\mathbf{x})
$$
where $\lambda_{\text{xy}}$ and $\lambda_{\text{goal}}$ are hyperparameters, $\Phi$ is kinematic function set.
## **2.4. Prediction: Occupancy Prediction**
OccFormer incorporate both **scene-level and agent-level semantics** in 2 aspects:
1. A **dense scene features** acquires agent-level features via an attention module when unrolling to future horizons.
2. **Instance-wise occupancy** is produced easily by a matrix multiplication between agent-level features and dense scene features.
OccFormer is composed of $T_o$ sequential blocks where $T_o$ indicates the prediction horizon. Each block takes as input the **rich agent features $G_t$** and **the state (dense feature) $F_{t−1}$**
from the previous layer. 
- **Rich agent features $G_t$:** is obtained with dynamics and spatial priors, previous motion queries are max pooled ($Q_{X} \in \mathbb{R}^{N_s\times D}$) then fused with upstream track query $Q_A$ and current position embedding $P_A$. The concatenated tensor is then passed through a **temporal-specific MLP**.
$$G^t=\text{MLP}_t(Concat(Q_A,P_A,Q_X)),\quad t=1,...,T_o$$
- **Dense scene features:** initially BEV feature $B$ is downscaled to 1/4 resolution to serve as first block input $F^0$. 
The architecture of OccFormer can be considered as 2 phases: **(1) capturing pixel-agent interaction and (2) predict instance-level occupancy**.
![[OccFormer.png|400]]
### Pixel-agent interaction
Each block receives the dense features $F^{t-1}$ as an input and feeds it to a downsample-upsample manner with an attention module in between to conduct pixel-agent interaction at 1/8 downscaled feature (denoted as $F^t_{ds}$). Attention module takes downscaled dense feature $F^t_{ds}$ as queries, agent features $G^t$ as keys and values. 
- Firstly, dense feature $F^t_{ds}$ is passed through self-attention layer to model responses between distant grids.
- Them, cross-attention layer to capture interactions between agent features $G^t$. This layer is constrained by an attention mask $O^t_m$ generated by multiplying an additional agent-level feature (also named as mask feature $M^t=MLP(G^t)$ which will be used for occupancy prediction) and dense feature $F^t_{ds}$, i.e. $O_m^{t}=M^t \cdot F^t_{ds}$.
$$D^t_{ds}=\text{MHCA}(\text{MHSA}(F^t_{ds}),G^t,\text{atten-mask}=O^t_m)$$
- Finally, $D_{ds}^t$ is upsampled to 1/4 size of $B$ and added with dense feature $F^{t-1}$ as a residual connection resulting feature $F^t$.
### Instance-level occupancy
Formally, in order to get an occupancy prediction of original size $H×W$ of BEV
feature $B$, the scene-level features $F^t$ are upsampled to $F^t_{dec}\in\mathbb{R}^{C \times H \times W}$ by a convolutional decoder. For the agent-level feature, the coarse mask feature $M^t$ is further updated to the occupancy feature $U^t \in \mathbb{R}^{N_a×C}$ by another MLP.
$$\hat{O}^t_A=U^t\cdot F^t_{dec}=\text{MLP}(M^t)\cdot F^t_{dec}$$
## **2.5 Planning**
Planner takes the ego-vehicle query generated from the tracking and motion forecasting
module. These two queries, along with the command embedding, are encoded. The BEV feature interaction module is built with standard transformer decoder layers. Then regress
the planning trajectory with MLP layers, which is denoted as $\hat{τ} \in \mathbb{R}^{T_p×2}$. Finally predicted occupancy $\hat{O}$ and trajectory $\hat{τ}$ are fed into the collision optimizer for obstacle avoidance.
$$τ^*=\text{argmin}_{τ}f(τ,\hat{τ},\hat{O})$$
where $\hat{τ}$ is the original planning prediction, $τ^*$ denotes the optimized planning which selected from multiple shooting trajectories $τ$ as to minimize cost function $f(.)$. $\hat{O}$ is a classical binary occupancy map merge from instance-wise occupancy prediction from OccFormer.
$$f(\tau, \hat{\tau}, \hat{\mathcal{O}}) = \lambda_{\text{coord}} \|\tau, \hat{\tau}\|_2 + \lambda_{\text{obs}} \sum_t \mathcal{D}(\tau_t, \hat{\mathcal{O}}^t)$$$$\mathcal{D}(\tau_t, \hat{\mathcal{O}}^t) = \sum_{(x, y) \in S} \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{\|\tau_t - (x, y)\|_2^2}{2\sigma^2}\right)$$![[Planner.png|400]]
## **2.6. Learning**
Firstly, jointly train perception parts, i.e., the tracking and mapping modules, for a few epochs (6), and then train the model end-to-end for rest epochs (20) with all perception, prediction and planning modules.
![[UniAD Learning | 1200]]
# **Notations**
![[Notations.png]]