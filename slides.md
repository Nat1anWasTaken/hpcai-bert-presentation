---
theme: ./theme
title: Fine-tuning BERT for tweets
class: text-start
drawings:
  persist: false
mdc: true
layout: cover
---

# Fine-Tuning BERT for Tweets
Achieving High Accuracy in Tweet Sentiment Classification

---
layout: default
---

# 原始的 Training Pipeline

- 使用 trainer 直接訓練 `google-bert/bert-base-uncased`
- 使用部分資料 (`train[:100]`、`eval[100:110]`)
- 只設定基本訓練參數：batch size、epoch、learning rate
- 僅採用 accuracy 作為評估指標
- 儲存 checkpoint 但無早停策略（early stopping）

---
layout: two-cols-header
---

# 我們的 Training Pipeline

::left::
- **資料集**
  - 使用完整 train/test 並切分 validation
- **評估方式**
  - 每 250 步評估一次 + 加入多指標 (acc/pr/rec/f1)
- **Loss 計算**
  - 加入 `compute_class_weight()` 解決不平衡問題
- **Early Stopping**
  - 使用 `EarlyStoppingCallback`
- **Learning rate**
  - 使用 `cosine scheduler` + warmup

::right::
- **Dropout**
  - 明確設定為 0.1，提升泛化能力
- **多 GPU 支援**
  - `torch.cuda.device_count()`、`gradient_accumulation`
- **批次大小與效能優化**
  - 多卡總 batch size 有效達到 1024，提升訓練效率
- **精度與記憶體**
  - 使用 `fp16`、梯度裁剪 (`max_grad_norm=1.0`)
- **Logging**
  - 更頻繁 `logging_steps=50` 且強化首次步紀錄

---
layout: cover
---

# The Actual Implementation

---
layout: code-right
---

# 資料分割
- 原始的設計只使用了`mteb/tweet_sentiment_extraction` 的
  <br>前 100 則資料
- 我們改使用了完整的資料集
- 以 9:1 的比例分割為訓練集與測試集

::code::
```py
from datasets import load_dataset

raw_dataset = load_dataset(
    "mteb/tweet_sentiment_extraction", 
    split="train"
)

train_size = int(0.9 * len(raw_dataset))
val_size = len(raw_dataset) - train_size

train_indices = list(range(train_size))
val_indices = list(
    range(train_size, len(raw_dataset))
)

train_dataset = raw_dataset.select(
    train_indices
)
val_dataset = raw_dataset.select(
    val_indices
)
```

---
layout: code-right
---

# 資料不平衡

- 推文資料類別不平衡
- 模型可能偏向出現次數多的類別
- 計算權重、加入交叉熵損失函數
- 公平學習每個類別

::code::

```py
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

labels = [
    example['label'] 
    for example in train_dataset
]

class_weights = compute_class_weight(
    'balanced', classes=[0,1,2], y=labels
)
class_weights_tensor = torch.tensor(
    class_weights, dtype=torch.float32
).to(device)

loss_fct = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)
```

---
layout: code-right
---

# Training Arguments

- 我們上網搜尋 BERT 的優化策略
- 適度的學習率、梯度累積、cosine 衰減排程、warmup 步數
- 提升訓練穩定性與收斂效果

::code::

```py
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=12,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    logging_dir="./logs",
    logging_steps=50,
    save_steps=250,
    eval_steps=250,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=4,
    dataloader_drop_last=True
)
```

---
layout: code-right
---

# Early Stopping 與 Checkpoints

為了防止過度訓練與資源浪費，我們加上 early stopping 當驗證集表現長時間無提升就自動停止訓練，並搭配定期儲存檢查點，保留最佳模型。

::code::

```py
from transformers import EarlyStoppingCallback

trainer = Trainer(...)
trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=5
    )
)
```

---
layout: code-right
---

# 多 GPU 設定與<br>有效批次計算

- 利用多 GPU 加速訓練
- 增加迭代速度

::code::

```py
import torch

device_count = torch.cuda.device_count()
per_device_batch_size = 64
accumulation_steps = 2
```

---
layout: default
---

# 訓練結果

| 指標      | 原版         | 改進版                        |
|-----------|--------------|-------------------------------|
| Accuracy  | 38% ~        | 78% ~                      |
| F1-score  | 無           | ↑ macro F1 顯著提升           |
| Precision | 無           | ↑                             |
| Recall    | 無           | ↑                             |

---
layout: cover
class: text-center
---

# 但是

---
layout: cover
class: text-center
---

# 80.15 %
<v-click>我們把我們所做的所有優化移除，只剩下使用完整的資料集、並提高訓練步數</v-click>

---
layout: default
class: overflow-y-scroll
---

```
[u9603854@un-ln01 BERT-Finetune-HPCAI-Camp]$ python ./run.py 
Starting automated BERT evaluation
Target accuracy: 79.95%
Strategy: Test checkpoints from smallest to largest, stop if accuracy drops

==================================================
INITIAL TRAINING
==================================================

=== Starting Training Iteration 1 ===
Training job submitted with ID: 752278
Waiting for training job 752278 to complete...
Job still running... checking again in 30 seconds
Job still running... checking again in 30 seconds
....
Training job completed!

==================================================
CHECKPOINT EVALUATION 1
==================================================
Found 7 checkpoints:
  - ./output_model/checkpoint-376 (step 376)
  - ./output_model/checkpoint-752 (step 752)
  - ./output_model/checkpoint-1128 (step 1128)
  - ./output_model/checkpoint-1504 (step 1504)
  - ./output_model/checkpoint-3008 (step 3008)
  - ./output_model/checkpoint-4512 (step 4512)
  - ./output_model/checkpoint-6016 (step 6016)
Running inference on ./output_model/checkpoint-376
Inference job submitted with ID: 752284
Waiting for training job 752284 to complete...
Job still running... checking again in 30 seconds
Job still running... checking again in 30 seconds
Training job completed!
Parsed accuracy: 78.4965034965035%

Checkpoint: ./output_model/checkpoint-376
Current accuracy: 78.4965034965035%
Previous accuracy: 0.0%
Target accuracy: 79.95%
Backed up best model to ./best_model_acc_78.50
Accuracy 78.4965034965035% is below target 79.95%
Testing next checkpoint...
```

---
layout: two-cols-header
---

<div>
  <h1>結果</h1>
</div>

::left::
<div class="h-full ">

<h2><code>bert-inf.err</code></h2>

```
...
loading miniconda3 with conda 24.5.0 and python 3.9
docs : https://hackmd.io/@kmo/twcc_hpc_conda
...

100%|██████████| 3432/3432 [00:18<00:00, 182.80it/s]
```
</div>

::right::
<h2><code>bert-inf.out</code></h2>
```
length of dataset is 3432
The generation accuracy is 80.15734265734265 %.
Total inference time is 18.775450706481934 sec.
```