import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_data_loaders
from model import DictionaryModel
from tqdm import tqdm

# Hyperparameters
vocab_size = 27  # BERT vocab size, since BERT uncased tokenizer was used
d_model = 256
nhead = 8
num_layers = 8
dim_feedforward = 256
learning_rate = 3e-4
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, test_loader):
    model = DictionaryModel(vocab_size, d_model, nhead, num_layers, dim_feedforward).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_ids = target_ids - 1036  # Convert token IDs to 0-26 range
            target_ids[target_ids < 0] = 0 #make sure padding tokens stay padding tokens
            optimizer.zero_grad()

            batch_size, target_len = target_ids.shape
            outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)

            # Initialize decoder input with the first token of input_ids
            decoder_input = input_ids[:, 0].unsqueeze(1)

            for t in range(target_len):
                # Generate a prediction for the next token
                output = model(decoder_input)
                
                outputs[:, t, :] = output[:, -1, :]
                
                # Use teacher forcing: next input is current target
                if t < target_len - 1:
                    decoder_input = torch.cat([decoder_input, target_ids[:, t].unsqueeze(1)], dim=1)
            
            loss = criterion(outputs.contiguous().view(-1, vocab_size), target_ids.contiguous().view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                batch_size, target_len = target_ids.shape
                outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)

                decoder_input = input_ids[:, 0].unsqueeze(1)
                target_ids = target_ids - 1036  # Convert token IDs to 0-26 range
                target_ids[target_ids < 0] = 0 #make sure padding tokens stay padding tokens
                for t in range(target_len):
                    output = model(decoder_input)
                    outputs[:, t, :] = output[:, -1, :]
                    
                    if t < target_len - 1:
                        # During evaluation, use the model's own predictions
                        _, next_token = output[:, -1, :].max(1)
                        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

                loss = criterion(outputs.contiguous().view(-1, vocab_size), target_ids.contiguous().view(-1))

                test_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('dictionary.json', batch_size=16)
    train(train_loader, test_loader)