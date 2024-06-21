import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class DictionaryModel(nn.Module):
    def __init__(self, hidden_size):
        super(DictionaryModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = hidden_size
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(self.bert.config.hidden_size, 27)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, word_end_pos):
        # Pass the input through BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Extract the word representations
        batch_size = input_ids.size(0)
        word_representations = []
        for i in range(batch_size):
            word_representations.append(last_hidden_state[i, 1:word_end_pos[i]])  # Exclude [CLS] token

        # Pad the word representations to the same length
        max_word_length = max(rep.size(0) for rep in word_representations)
        padded_word_representations = torch.zeros(batch_size, max_word_length, self.bert.config.hidden_size, device=input_ids.device)
        for i, rep in enumerate(word_representations):
            padded_word_representations[i, :rep.size(0)] = rep

        # Apply the final linear layer and softmax
        output = self.fc(padded_word_representations)
        output = self.softmax(output)

        return output