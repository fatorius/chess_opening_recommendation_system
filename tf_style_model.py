import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
EMBEDDING_DIM = 32
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

class ChessDataset(Dataset):
    """Dataset for single-player proxy training samples"""
    def __init__(self, players, openings, results, ratings):
        if ratings is None:
            raise ValueError("ratings are required for this model.")
        self.players = players
        self.openings = openings
        self.results = results
        self.ratings = ratings
        
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return {
            'player': self.players[idx],
            'opening': self.openings[idx],
            'rating': self.ratings[idx],
            'result': self.results[idx]
        }

class TFStyleChessModel(nn.Module):
    """TensorFlow-style architecture in PyTorch"""
    def __init__(self, num_players, num_openings, embedding_dim=4):
        super().__init__()
        
        # Embedding layers
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.opening_embedding = nn.Embedding(num_openings, embedding_dim)
        
        # Input: opening_emb(4) + player_emb(4) + rating(1)
        concat_dim = embedding_dim * 2 + 1  # 4 + 4 + 1 = 9
        self.dense = nn.Linear(concat_dim, 3)  # loss/draw/win logits
    
    def forward(self, opening_input, player_input, rating_input):
        if rating_input is None:
            raise ValueError("rating_input is required.")

        # Embeddings: [B, 1] -> [B, 1, emb_dim] -> flatten to [B, emb_dim]
        opening_vec = self.opening_embedding(opening_input).squeeze(1)          # [B, 4]
        player_vec = self.player_embedding(player_input).squeeze(1)             # [B, 4]
        rating_input = rating_input.view(-1, 1)
        
        # Concatenate all features
        x = torch.cat([
            opening_vec,
            player_vec, 
            rating_input
        ], dim=1)  # [B, 9]
        
        return self.dense(x)

def collate_fn(batch):
    """Custom collate function"""
    players = torch.tensor([b['player'] for b in batch], dtype=torch.long).unsqueeze(1)
    openings = torch.tensor([b['opening'] for b in batch], dtype=torch.long).unsqueeze(1)
    ratings = torch.tensor([b['rating'] for b in batch], dtype=torch.float32).unsqueeze(1)
    results = torch.tensor([b['result'] for b in batch], dtype=torch.long)
    
    return openings, players, ratings, results

def prepare_data(csv_path, test_size=0.2):
    """Load and prepare data"""
    df = pd.read_csv(csv_path)
    
    # Convert result_code to player-perspective class labels: 0=loss, 1=draw, 2=win
    white_result_map = {0: 0, 1: 2, 2: 1}  # black win=loss, white win=win, draw=draw
    black_result_map = {0: 2, 1: 0, 2: 1}  # black win=win, white win=loss, draw=draw
    df['white_result'] = df['result_code'].map(white_result_map)
    df['black_result'] = df['result_code'].map(black_result_map)
    
    # Normalize ratings to [0, 1] using the same scale for both colors
    rating_min = min(df['white_rating'].min(), df['black_rating'].min())
    rating_max = max(df['white_rating'].max(), df['black_rating'].max())
    
    df['white_rating_norm'] = (df['white_rating'] - rating_min) / (rating_max - rating_min)
    df['black_rating_norm'] = (df['black_rating'] - rating_min) / (rating_max - rating_min)

    # Expand each game into two samples: one for white, one for black
    white_samples = df[['white_player_id', 'opening_id', 'white_rating_norm', 'white_result']].copy()
    white_samples.columns = ['player_id', 'opening_id', 'rating_norm', 'result']

    black_samples = df[['black_player_id', 'opening_id', 'black_rating_norm', 'black_result']].copy()
    black_samples.columns = ['player_id', 'opening_id', 'rating_norm', 'result']

    samples = pd.concat([white_samples, black_samples], ignore_index=True)
    
    # Get embedding dimensions
    num_players = max(df['white_player_id'].max(), df['black_player_id'].max()) + 1
    num_openings = df['opening_id'].max() + 1
    
    # Split data
    train_idx, val_idx = train_test_split(
        np.arange(len(samples)), 
        test_size=test_size, 
        random_state=42
    )
    
    train_df = samples.iloc[train_idx].reset_index(drop=True)
    val_df = samples.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    print(f"Players: {num_players}, Openings: {num_openings}")
    
    return train_df, val_df, num_players, num_openings

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for openings, players, ratings, results in train_loader:
        openings = openings.to(device)
        players = players.to(device)
        ratings = ratings.to(device)
        results = results.to(device)
        
        optimizer.zero_grad()
        outputs = model(openings, players, ratings)
        loss = criterion(outputs, results)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for openings, players, ratings, results in val_loader:
            openings = openings.to(device)
            players = players.to(device)
            ratings = ratings.to(device)
            results = results.to(device)
            
            outputs = model(openings, players, ratings)
            loss = criterion(outputs, results)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Prepare data
    train_df, val_df, num_players, num_openings = prepare_data('games_encoded.csv')
    
    # Create datasets
    train_dataset = ChessDataset(
        train_df['player_id'].values,
        train_df['opening_id'].values,
        train_df['result'].values,
        train_df['rating_norm'].values
    )
    
    val_dataset = ChessDataset(
        val_df['player_id'].values,
        val_df['opening_id'].values,
        val_df['result'].values,
        val_df['rating_norm'].values
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = TFStyleChessModel(
        num_players=num_players,
        num_openings=num_openings,
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tf_style_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    # Load and save best model
    model.load_state_dict(torch.load('best_tf_style_model.pth'))
    torch.save(model.state_dict(), 'tf_style_model.pth')
    print(f"✓ Model saved to tf_style_model.pth")

if __name__ == '__main__':
    main()
