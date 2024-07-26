from transformers import AutoTokenizer, BartForConditionalGeneration, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 初始化tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"instruction: {item['instruction']} input: {item['input']}"
        target_text = item['output']

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 准备数据
data = [
    {
        'instruction': '2023年第1季度，600372.SH的营业成本是多少？',
        'input': '',
        'output': '2023年第1季度，600372.SH的营业成本为2233536850.92元。'
    }
    # 添加更多数据...
]

# 创建数据集和数据加载器
dataset = CustomDataset(data, tokenizer, max_length=64)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
num_epochs = 1
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 保存模型
# model.save_pretrained("./fine_tuned_bart_model")
# tokenizer.save_pretrained("./fine_tuned_bart_model")

print("Training completed and model saved.")