import json
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# Initialize tokenizer outside the class
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DictionaryDataset(Dataset):
    def __init__(self, json_file, mask_ratio=0.33):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.words = list(self.data.keys())
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        description = self.data[word]

        # Randomly mask 33% of characters
        masked_word = self.mask_word(word)
        

        # Tokenize the masked word + description
        input_text = f"{masked_word} {description}"
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=512)

        # Tokenize the original word (target)
        word_chars=[tokenizer.encode(char, add_special_tokens=False) for char in word]
        word_chars_ids = [item for sublist in word_chars for item in sublist]
        target_ids = torch.tensor(word_chars_ids)
        #target_ids = tokenizer.encode(word, add_special_tokens=True)

        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }

    def mask_word(self, word):
        chars = list(word)
        num_to_mask = max(1, int(len(chars) * self.mask_ratio))
        mask_indices = random.sample(range(len(chars)), num_to_mask)
        for idx in mask_indices:
            chars[idx] = '_'
        return ''.join(chars)

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = pad_sequence([item['target_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids
    }

def get_data_loaders(json_file, batch_size=16, test_size=0.2):
    dataset = DictionaryDataset(json_file)
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
        print(batch['input_ids'].shape, batch['target_ids'].shape)
        print(tokenizer.decode(102))
        break
    print("Test loader:")
    for batch in test_loader:
        print(batch['input_ids'].shape, batch['target_ids'].shape)
        break