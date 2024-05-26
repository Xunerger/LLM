import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4  # How many batches per training step
context_length = 16  # Length of the token chunk each batch
d_model = 64  # The size of our model token embeddings
num_blocks = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi_head attention
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 5000  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Get the dataset
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# Tokenize the text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1

# Split init train validation
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]


class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.register_buffer('mask', torch.tril(
            torch.ones(context_length, context_length)))

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attention = Q @ K.transpose(-2, -1) / math.sqrt(d_model)
        attention = attention.masked_fill(
            self.mask[:attention.size(-2), :attention.size(-1)] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        output = attention @ V
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            [ScaledDotProductAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model * num_heads, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        projected = self.projection_layer(concatenated)
        output = self.dropout(projected)
        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedforwardNetwork()

    def forward(self, x):
        attention_output = self.multi_head_attention(x)
        x = x + attention_output
        x = self.layer_norm1(x)
        feedforward_output = self.feedforward_network(x)
        x = x + feedforward_output
        x = self.layer_norm2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_lookup_table = nn.Embedding(
            max_token_value, d_model)
        self.position_encoding_lookup_table = self.generate_positional_encoding(
            context_length, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(num_blocks)])
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def generate_positional_encoding(self, max_len, d_model):
        position_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_lookup_table(idx)
        position_embeddings = self.position_encoding_lookup_table[:T, :].to(
            device)
        x = token_embeddings + position_embeddings
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # view()用于改变tensor的形状，数值不改变
            targets = targets.view(B * T)
            # cross_entropy()用于求交叉熵https://blog.csdn.net/wuliBob/article/details/104119616
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_length:]
            logits, _ = self.forward(idx_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = Model().to(device)

data = torch.tensor(tokenized_text, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(data, batch_size):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, batch_size)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print(f"step {iter}: loss {loss:.4f}")

    xb, yb = get_batch(train_data, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoding.decode(model.generate(context, max_new_tokens=200)[0].tolist()))

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
model.eval()
start = 'The product is'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
