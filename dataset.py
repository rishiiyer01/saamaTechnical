import json
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# Initialize tokenizer outside the class to avoid reloading it multiple times
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DictionaryDataset(Dataset):
    def __init__(self, json_file, mask_prob=0.15):
        with open(json_file, 'r') as file:
            data = json.load(file)
            self.entries = []
            for word, description in data.items():
                word_chars = [tokenizer.encode(char, add_special_tokens=False)[0] for char in word]
                combined_input = ' '.join([tokenizer.decode([char_id]) for char_id in word_chars]) + " [SEP] " + description
                combined_tokens = tokenizer.encode(combined_input, add_special_tokens=True, truncation=True, max_length=512)
                word_end_pos = len(word_chars) + 1  # +1 for [CLS] token
                self.entries.append({
                    'combined_tokens': combined_tokens,
                    'word_end_pos': word_end_pos,
                    'full_word': word
                })
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        combined_tokens = entry['combined_tokens']
        word_end_pos = entry['word_end_pos']
        
        # Apply random masking to the word part
        masked_tokens = combined_tokens.copy()
        for i in range(1, word_end_pos):  # Start from 1 to skip [CLS]
            if random.random() < self.mask_prob:
                masked_tokens[i] = tokenizer.mask_token_id
        
        return {
            'input_ids': torch.tensor(masked_tokens),
            'attention_mask': torch.ones(len(masked_tokens), dtype=torch.long),
            'word_end_pos': word_end_pos,
            'full_word': entry['full_word']
        }

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    word_end_pos = torch.tensor([item['word_end_pos'] for item in batch])
    full_words = [item['full_word'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_end_pos': word_end_pos,
        'full_words': full_words
    }

def get_data_loaders(json_file, batch_size=16, test_size=0.2, mask_prob=0.15):
    dataset = DictionaryDataset(json_file, mask_prob)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    return train_loader, test_loader

# for testing
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('dictionary.json')
    print("Train loader:")
    for batch in train_loader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Word end positions: {batch['word_end_pos']}")
        print(f"Full words: {batch['full_words']}")
        break
    print("\nTest loader:")
    for batch in test_loader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Word end positions: {batch['word_end_pos']}")
        print(f"Full words: {batch['full_words']}")
        break