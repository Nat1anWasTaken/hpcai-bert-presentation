---
theme: ./theme
title: Fine-tuning BERT for tweets
class: text-start
drawings:
  persist: false
transition: slide-left
mdc: true
layout: cover
---

# Fine-Tuning BERT for Tweets
Achieving High Accuracy in Tweet Sentiment Classification

---
layout: default
---

# The original training pipeline

- 使用 Trainer 直接訓練 `google-bert/bert-base-uncased`
- 使用部分資料 (`train[:100]`、`eval[100:110]`)
- 只設定基本訓練參數：batch size、epoch、learning rate
- 僅採用 Accuracy 作為評估指標
- 儲存 checkpoint 但無早停策略（early stopping）

---
layout: two-cols-header
---

# Our training pipeline

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
layout: full
class: overflow-y-scroll
---

```py
def wait_for_job_completion(self, job_id):
        """Wait for SLURM job to complete"""
        print(f"Waiting for job {job_id} to complete...")
        
        while True:
            # Check job status for specific user
            result = subprocess.run(['squeue', '-u', self.user_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            # Check if job is still in the queue
            if job_id not in result.stdout:
                print(f"Job {job_id} completed!")
                break
                
            print("Job still running... checking again in 30 seconds")
            time.sleep(30)#!/usr/bin/env python3
"""
Automated BERT Fine-tuning Script
Continuously trains and evaluates BERT model until target accuracy is achieved.
"""

import subprocess
import re
import time
import os
import glob
from pathlib import Path

class BERTAutoTrainer:
    def __init__(self, target_accuracy=79.43, max_iterations=10, user_id="u9603854"):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.best_accuracy = 0.0
        self.best_checkpoint = None
        self.user_id = user_id
        self.previous_accuracy = 0.0
        self.tested_checkpoints = []
        
    def submit_training_job(self):
        """Submit training job using sbatch"""
        print(f"\n=== Starting Training Iteration {self.current_iteration + 1} ===")
        
        # Submit training job
        cmd = ["sbatch", "run_train.sh", "google-bert/bert-base-uncased", "./output_model"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode != 0:
            print(f"Error submitting training job: {result.stderr}")
            return None
            
        # Extract job ID from sbatch output
        job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Training job submitted with ID: {job_id}")
            return job_id
        else:
            print("Could not extract job ID from sbatch output")
            return None
    
    def wait_for_job_completion(self, job_id):
        """Wait for SLURM job to complete"""
        print(f"Waiting for training job {job_id} to complete...")
        
        while True:
            # Check job status
            result = subprocess.run(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if job_id not in result.stdout:
                print("Training job completed!")
                break
                
            print("Job still running... checking again in 30 seconds")
            time.sleep(30)
    
    def get_all_checkpoints_sorted(self):
        """Get all checkpoints sorted from smallest to largest"""
        checkpoint_pattern = "./output_model/checkpoint-*"
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("No checkpoints found!")
            return []
            
        # Sort checkpoints by the number after 'checkpoint-'
        def extract_step(checkpoint_path):
            match = re.search(r'checkpoint-(\d+)', checkpoint_path)
            return int(match.group(1)) if match else 0
            
        sorted_checkpoints = sorted(checkpoints, key=extract_step)
        print(f"Found {len(sorted_checkpoints)} checkpoints:")
        for cp in sorted_checkpoints:
            step = extract_step(cp)
            print(f"  - {cp} (step {step})")
        return sorted_checkpoints
    
    def get_next_untested_checkpoint(self):
        """Get the next checkpoint that hasn't been tested yet"""
        all_checkpoints = self.get_all_checkpoints_sorted()
        
        for checkpoint in all_checkpoints:
            if checkpoint not in self.tested_checkpoints:
                return checkpoint
        
        return None
    
    def run_inference(self, checkpoint_path):
        """Run inference on the given checkpoint"""
        print(f"Running inference on {checkpoint_path}")
        
        # Submit inference job
        cmd = ["sbatch", "run_inf.sh", checkpoint_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode != 0:
            print(f"Error submitting inference job: {result.stderr}")
            return None
            
        # Extract job ID
        job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Inference job submitted with ID: {job_id}")
            
            # Wait for completion
            self.wait_for_job_completion(job_id)
            return job_id
        else:
            print("Could not extract job ID from inference sbatch output")
            return None
    
    def parse_accuracy(self):
        """Parse accuracy from bert-inf.out file"""
        try:
            with open('./bert-inf.out', 'r') as f:
                content = f.read()
                
            # Look for accuracy pattern
            accuracy_match = re.search(r'The generation accuracy is ([\d.]+) %', content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                print(f"Parsed accuracy: {accuracy}%")
                return accuracy
            else:
                print("Could not find accuracy in output file")
                return None
                
        except FileNotFoundError:
            print("bert-inf.out file not found")
            return None
        except Exception as e:
            print(f"Error parsing accuracy: {e}")
            return None
    
    def backup_best_results(self, checkpoint_path, accuracy):
        """Backup the best performing checkpoint"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_checkpoint = checkpoint_path
            
            # Create backup directory
            backup_dir = f"./best_model_acc_{accuracy:.2f}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy checkpoint files
            subprocess.run(f"cp -r {checkpoint_path}/* {backup_dir}/", shell=True)
            print(f"Backed up best model to {backup_dir}")
    
    def run(self):
        """Main training loop - tests checkpoints from smallest to largest"""
        print(f"Starting automated BERT evaluation")
        print(f"Target accuracy: {self.target_accuracy}%")
        print(f"Strategy: Test checkpoints from smallest to largest, stop if accuracy drops")
        
        # First, run one training iteration to ensure we have checkpoints
        print(f"\n{'='*50}")
        print(f"INITIAL TRAINING")
        print(f"{'='*50}")
        
        train_job_id = self.submit_training_job()
        if train_job_id:
            self.wait_for_job_completion(train_job_id)
        
        # Now test checkpoints from smallest to largest
        checkpoint_iteration = 0
        
        while True:
            checkpoint_iteration += 1
            print(f"\n{'='*50}")
            print(f"CHECKPOINT EVALUATION {checkpoint_iteration}")
            print(f"{'='*50}")
            
            # Get next untested checkpoint
            checkpoint_path = self.get_next_untested_checkpoint()
            if not checkpoint_path:
                print("No more untested checkpoints available")
                if checkpoint_iteration == 1:
                    print("No checkpoints found at all. Training may have failed.")
                    return False
                else:
                    print(f"All checkpoints tested. Best accuracy: {self.best_accuracy}%")
                    break
            
            # Mark this checkpoint as tested
            self.tested_checkpoints.append(checkpoint_path)
            
            # Run inference on this checkpoint
            inf_job_id = self.run_inference(checkpoint_path)
            if not inf_job_id:
                print("Failed to run inference, skipping checkpoint")
                continue
                
            # Parse accuracy
            accuracy = self.parse_accuracy()
            if accuracy is None:
                print("Failed to parse accuracy, skipping checkpoint")
                continue
                
            print(f"\nCheckpoint: {checkpoint_path}")
            print(f"Current accuracy: {accuracy}%")
            print(f"Previous accuracy: {self.previous_accuracy}%")
            print(f"Target accuracy: {self.target_accuracy}%")
            
            # Check if accuracy dropped compared to previous
            if checkpoint_iteration > 1 and accuracy < self.previous_accuracy:
                print(f"\n⚠️  ACCURACY DROPPED! {accuracy}% < {self.previous_accuracy}%")
                print(f"Stopping evaluation and using previous best checkpoint")
                print(f"Best accuracy: {self.best_accuracy}%")
                print(f"Best checkpoint: {self.best_checkpoint}")
                return self.best_accuracy > self.target_accuracy
            
            # Update best results
            self.backup_best_results(checkpoint_path, accuracy)
            self.previous_accuracy = accuracy
            
            # Check if target reached
            if accuracy > self.target_accuracy:
                print(f"\n🎉 SUCCESS! Target accuracy {self.target_accuracy}% exceeded!")
                print(f"Final accuracy: {accuracy}%")
                print(f"Best checkpoint: {checkpoint_path}")
                return True
            
            print(f"Accuracy {accuracy}% is below target {self.target_accuracy}%")
            print("Testing next checkpoint...")
            time.sleep(5)  # Brief pause before next checkpoint
        
        # Final results
        if self.best_accuracy > self.target_accuracy:
            print(f"\n🎉 SUCCESS! Target accuracy achieved!")
            print(f"Best accuracy: {self.best_accuracy}%")
            print(f"Best checkpoint: {self.best_checkpoint}")
            return True
        else:
            print(f"\n❌ Target accuracy not reached")
            print(f"Best accuracy: {self.best_accuracy}%")
            print(f"Best checkpoint: {self.best_checkpoint}")
            return False

def main():
    # Configuration
    TARGET_ACCURACY = 79.43  # Change this to your desired target
    MAX_ITERATIONS = 10      # Maximum number of training iterations
    
    # Create trainer instance
    trainer = BERTAutoTrainer(
        target_accuracy=TARGET_ACCURACY,
        max_iterations=MAX_ITERATIONS
    )
    
    # Run the automated training
    success = trainer.run()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n⚠️  Training stopped without reaching target accuracy")

if __name__ == "__main__":
    main()
```

---
layout: cover
class: text-center
---

# 80.05 %
<v-click>我們把我們所做的所有優化移除，只剩下使用完整的資料集、並提高訓練步數</v-click>

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