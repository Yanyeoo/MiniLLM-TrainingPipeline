# LLM+多模态 轻量对话模型全流程训练与对齐（Pretrain → SFT → RLHF）
## 🏥 MedicalRobot-MiniLLM  
轻量医疗对话模型全流程训练与对齐（Pretrain → SFT → RLHF）

---

### 📌 项目简介
本项目面向医疗机器人问诊与导诊场景，构建一个**轻量级医疗对话模型**，重点提升：

- 医疗问答的**一致性与可靠性**
- 多轮对话的**稳定性**
- 小模型的**推理能力与安全性**

支持从 **预训练 → 指令微调 → 推理增强 → 偏好对齐（RLHF/DPO）** 的完整训练链路。

---

### ⚙️ 仅针对本地环境配置 从0实现详见单一项目
```bash
### CUDA 12.1

export PATH=/opt/data/jiayan/envs/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/data/jiayan/envs/cuda-12.1/lib64:$LD_LIBRARY_PATH

nvcc -V


### Conda 环境

export MY_HOME="/opt/data/jiayan"
source $MY_HOME/envs/miniconda3/bin/activate minimind
```



### 📂 数据说明

⚠️ 数据集未上传

* 医疗问答数据（SFT）
* 推理链数据（CoT / reasoning）
* 偏好对齐数据（DPO / RLHF）

👉 具体参考项目 `README` 说明

---

### 🚀 训练流程记录

#### 1️⃣ 预训练（Pretrain）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_pretrain.py
```

---

#### 2️⃣ mini_512有监督微调（SFT）

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_full_sft.py
```

---

#### 3️⃣ 1024知识蒸馏知识蒸馏（Distillation）

基于高质量模型输出数据进行黑盒蒸馏：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_full_sft.py
```

---

#### 4️⃣ LoRA 微调（高效适配）

```bash
torchrun --nproc_per_node=1 train_lora.py
```

📌 数据：医学领域智能引导问答数据集

---

### 🧠 方法亮点

* ✅ 小模型完整训练链路（Pretrain → SFT → RLHF）
* ✅ 引入多步推理数据，提升诊断解释能力
* ✅ 使用 DPO / 偏好优化降低 hallucination
* ✅ 面向医疗场景的安全对齐设计

---

### 🔮 扩展方向

* 多模态医疗对话（Vision + Language）
* 医疗机器人真实部署（边缘设备优化）
* 与自动驾驶 VLA 模型结构对齐（跨领域迁移）

---

## 🚗 Multi-Modal Cockpit Dialogue（扩展项目）

轻量多模态座舱对话模型（语音 / 视觉 / 指令理解）

👉 正在开发中...


## 👉 Demo / Results
```markdown
## 📊 Results
| Model | Accuracy | Hallucination ↓ | Stability ↑ |
|------|--------|----------------|-------------|
| Base | xx | xx | xx |
| Ours | xx | xx | xx |
````
