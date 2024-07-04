import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertTokenizer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        qkv=x.transpose(0, 1)  # Transpose to (seq_len, batch_size, hidden_size)
        attn_output, _ = self.self_attn(qkv, qkv, qkv, attn_mask=mask) #query, key, value tensors are the same
        attn_output=attn_output.transpose(0, 1)  # Transpose to (batch_size,seq_len, hidden_size)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DictionaryModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(DictionaryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj=nn.Linear(self.bert.config.hidden_size, d_model)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.GELU=nn.GELU()
        
    def forward(self, x):
        attention_mask=x.ne(0)
        berted = self.bert(
            x,
            attention_mask=attention_mask,
            token_type_ids=None  # Disable token type embeddings
        )
        #x = self.pos_encoding(x)
        x=berted.last_hidden_state
        xembedding=self.proj(x)
        x=self.GELU(xembedding)
        mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        for layer in self.layers:
            x = layer(x, mask)
        x=x+xembedding
        output = self.fc_out(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask