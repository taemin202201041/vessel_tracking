**CRACK DETECTION AND REFINEMENT VIA DEEP REINFORCEMENT LEARNIN**
1. 2단계 하이브리드 프레임워크 (Two-Stage Hybrid Framework)일반적인 Deep Learning 모델의 한계를 RL로 보완하는 구조입니다.Stage 1: \(U-Net\) (ResNet-50 backbone)을 사용하여 이미지 전체에서 균열(Crack)의 대략적인 위치를 파악하고 초기 예측 지도(Initial Prediction)를 생성합니다.Stage 2: RefineNet이라 불리는 RL 에이전트가 초기 예측 결과의 오류(끊긴 부분, 모호한 경계 등)를 반복적으로 수정합니다.2. 픽셀 단위 멀티 에이전트 강화 학습 (Multi-agent RL per Pixel)이미지의 각 픽셀을 하나의 독립적인 RL 에이전트로 간주하는 독특한 방식을 취합니다.각 픽셀 에이전트는 자신의 상태를 관찰하고 행동을 결정합니다.\(CNN\) 구조를 공유하므로, 인접한 픽셀 에이전트들이 서로의 상태 정보를 참조할 수 있어 국부적인 특징뿐만 아니라 주변 맥락까지 고려한 정제가 가능합니다.3. A3C (Asynchronous Advantage Actor-Critic) 알고리즘 활용학습의 효율성과 안정성을 위해 \(A3C\) 알고리즘을 채택했습니다.Policy Head (\(\pi\)): 각 픽셀에서 취할 최적의 행동(확률 조정)을 결정합니다.Value Head (\(V\)): 현재 상태의 가치를 평가하여 학습의 분산을 줄입니다.네트워크 업데이트는 다음과 같은 경사 하강법 식을 따릅니다:
\[
d\theta = \sum_{t=1}^{T} \left[ \nabla_{\theta} \log \pi(a_i^{(t)} | s_i^{(t)}; \theta) (R_i^{(t)} - V(s_i^{(t)}; \theta_v)) \right]
\]4. 정밀하게 설계된 MDP (State, Action, Reward)RL의 핵심 구성 요소를 세그멘테이션 정제 목적에 맞게 설계했습니다.상태 (State, \(s_i^{(t)}\)): 원본 이미지 픽셀(\(x_i\)), 현재 세그멘테이션 확률(\(p_i^{(t)}\)), 그리고 이전 행동 이력(\(h_i^{(t)}\))을 결합하여 구성합니다.행동 (Action, \(a_i^{(t)}\)): 픽셀이 균열일 확률을 직접 조정하는 이산적인 값들의 집합입니다.\(A = \{ \pm 0.1, \pm 0.2, \pm 0.4 \}\)업데이트 규칙: \(p_i^{(t+1)} = \text{clip}(p_i^{(t)} + a_i^{(t)}, 0, 1)\)보상 함수 (Reward Function, \(r_i^{(t)}\)): 정답(\(Ground Truth\))과의 교차 엔트로피(Cross Entropy, CE) 개선량을 보상으로 줍니다. 즉, 예측이 정확해질수록 더 큰 보상을 받습니다.
\[
r_i^{(t)} = CE(p_i^{(t)}, y_i) - CE(p_i^{(t+1)}, y_i)
\]5. 반복적 정제 프로세스 (Iterative Refinement)한 번의 예측으로 끝내는 것이 아니라, 여러 타임스텝(\(T=5\)) 동안 반복적으로 예측치를 수정합니다.이 과정에서 에이전트는 기존에 발견된 균열을 'Prior(사전 정보)'로 활용하여, 거기서부터 가지를 뻗어나가듯(branch out) 누락된 균열을 찾아내거나 끊어진 틈을 메웁니다.실험 결과, 반복 횟수가 늘어날수록 성능이 점진적으로 향상되다가 포화(Saturation)되는 것을 확인했습니다.


**3D Vessel Centerline Extraction via VascularMorphology Driven Adaptive DeepReinforcement Tracking**
[핵심 기술 1] 혈관 스케일 인식 기반 적응형 상태 (Adaptive State with VSP)에이전트의 시야(Receptive Field)를 혈관 굵기에 따라 동적으로 조절하여 작은 혈관에서의 정확도를 높이는 기술입니다.상태 정의 (\(s_t\)):
\[s_t = \{p_t, w_t\}\]
(\(p_t\): 에이전트의 현재 위치 좌표, \(w_t\): 현재 탐색 영역의 크기/Receptive Field)Vascular Scale Perception (VSP) 모델:입력: \(p_t\)를 중심으로 하는 \(w_t \times w_t \times w_t\) 크기의 3D 패치 (32x32x32로 정규화).출력: 스케일링 인자 \(g = VS_{t-1} / VS_t\) (\(VS\)는 혈관 스케일/지름).업데이트 규칙:
\[w_{t+1} = w_t / g\]
(혈관이 얇아지면 \(w\)를 줄여 미세 구조에 집중하고, 굵어지면 \(w\)를 키워 주변 맥락을 파악함)[핵심 기술 2] 적응형 딥 강화 추적기 구조 (Tracker Architecture)복잡한 혈관 윤곽과 특징을 효과적으로 추출하기 위한 신경망 설계 기술입니다.모델 구조: Dueling DQN 기반 아키텍처.주요 컴포넌트:Residual Connections: 경사 소실 방지 및 빠른 수렴.Dilated Convolutions (확장 컨볼루션): 파라미터 증가 없이 수용 영역(Receptive Field)을 확장하여 대동맥과 같은 큰 구조의 정보를 포착.Squeeze-and-Excitation (SE) Modules: 채널 어텐션을 통해 중요한 특징 채널의 가중치를 높이고 불필요한 정보를 억제.입력 및 출력:입력: \(w_t\) 크기의 정규화된 3D 패치 (23x23x23).출력: 100개의 사전 정의된 방향 벡터 중 가장 높은 가치를 가진 액션 \(a_t\) (단위 접선 벡터).[핵심 기술 3] 에이전트 이동 및 상태 전이 로직 (Agent Navigation)추적 프로세스를 물리적으로 실행하는 수식입니다.이동 수식:
\[p_{t+1} = p_t + a_t \cdot w_t \cdot \alpha\]
(\(\alpha\): 스텝 크기 계수, 논문에서는 \(1/3\) 사용)종료 조건 (Termination Rules):현재 위치 \(p_t\)가 Ground Truth 중심선에서 6 voxels 이상 벗어날 때.이미지 경계에 도달할 때.최대 스텝 수(1,000 steps)에 도달할 때.[핵심 기술 4] 혈관 형태 기반 보상 함수 (Morphology-based Reward)에이전트가 중심선을 유지하면서 전진하도록 유도하는 강화학습 핵심 알고리즘입니다.최종 보상 (\(r_t\)):
\[r_t = \beta r_1 + (1 - \beta)r_2\]
(논문에서는 \(\beta = 0.9\) 설정)근접성 보상 (\(r_1\)):
\[r_1 = \begin{cases} \frac{1}{1 + \exp((d_1 + d_3)/2)} + \frac{1}{2\lambda_t}(d_2 - d_3) & \text{if } d_2 - d_3 > 0 \\ 0 & \text{if } d_2 - d_3 \le 0 \end{cases}\]
(\(d_n\): 예측 지점과 실제 중심선 간의 거리들, \(\lambda_t\): 스텝 크기 \(w_t \cdot \alpha\))정방향 및 복귀 보상 (\(r_2\)):
\[r_2 = \begin{cases} \frac{1}{2\lambda_t}(d_2 - d_4) & \text{if } d_1 \le l_s \\ \frac{1}{2\lambda_t}(d_1 - d_3) & \text{if } d_1 > l_s \end{cases}\]
(\(l_s\): 1 voxel, 중심선 이탈 판단 기준)전략: 중심선에 잘 붙어 있을 때는(\(d_1 \le l_s\)) 전진을 독려하고, 벗어났을 때는(\(d_1 > l_s\)) 다시 중심선으로 끌어당기는 역할을 함.[핵심 기술 5] 모델 최적화 및 학습 전략 (Optimization)Loss Function (Huber Loss 기반 MSE):
\[L = \frac{1}{B} \sum_{i=1}^{B} (Q(s_t, a_t | \theta) - y_t)^2\]Target Q-value (\(y_t\)):
\[y_t = r_t + \gamma \max_{a'} Q'(s', a'; \theta')\]
(\(\gamma = 0.9\): 감쇄 계수)경험 재생 버퍼 (Replay Buffer): 크기 8,000의 버퍼를 사용하여 \((s_t, a_t, r_t, s_{t+1})\) 쌍을 저장하고 배치 크기 64로 무작위 샘플링하여 학습.