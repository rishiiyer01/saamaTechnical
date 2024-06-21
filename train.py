import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_data_loaders
from model import DictionaryModel
from tqdm import tqdm

# Hyperparameters
hidden_size = 256  
learning_rate = 1e-3
epochs = 10
mask_prob = 0.33

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, test_loader):
    model = DictionaryModel(hidden_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_end_pos = batch['word_end_pos'].to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_mask, word_end_pos)

            # Prepare the target: the original (unmasked) word characters
            target = input_ids[:, 1:max(word_end_pos)].clone()  # Exclude [CLS] token
            target[target == 0] = -100  # Ignore padding in loss calculation

            loss = criterion(output.view(-1, 27), target.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                word_end_pos = batch['word_end_pos'].to(device)

                output = model(input_ids, attention_mask, word_end_pos)

                target = input_ids[:, 1:max(word_end_pos)].clone()
                target[target == 0] = -100

                loss = criterion(output.view(-1, 27), target.view(-1))
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('dictionary.json', batch_size=16, mask_prob=mask_prob)
    train(train_loader, test_loader)