"""Train player + opening embeddings via implicit-feedback matrix factorization.

Proxy task: given (player_id, opening_id), did this player play this opening?
Positives are observed games (each game contributes one row per color).
Negatives are random openings paired with the same player.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

EMBEDDING_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-6
NEG_PER_POS = 4
VAL_SPLIT = 0.1


class PositivesDataset(Dataset):
    def __init__(self, players: np.ndarray, openings: np.ndarray):
        self.players = torch.as_tensor(np.ascontiguousarray(players), dtype=torch.long)
        self.openings = torch.as_tensor(np.ascontiguousarray(openings), dtype=torch.long)

    def __len__(self) -> int:
        return self.players.shape[0]

    def __getitem__(self, idx: int):
        return self.players[idx], self.openings[idx]


class MFModel(nn.Module):
    def __init__(self, num_players: int, num_openings: int, dim: int):
        super().__init__()
        self.player_emb = nn.Embedding(num_players, dim)
        self.opening_emb = nn.Embedding(num_openings, dim)
        self.opening_bias = nn.Embedding(num_openings, 1)
        nn.init.normal_(self.player_emb.weight, std=0.05)
        nn.init.normal_(self.opening_emb.weight, std=0.05)
        nn.init.zeros_(self.opening_bias.weight)

    def forward(self, player_ids: torch.Tensor, opening_ids: torch.Tensor) -> torch.Tensor:
        p = self.player_emb(player_ids)
        o = self.opening_emb(opening_ids)
        b = self.opening_bias(opening_ids).squeeze(-1)
        return (p * o).sum(-1) + b


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    white = df[['white_player_id', 'opening_id']].rename(
        columns={'white_player_id': 'player_id'}
    )
    black = df[['black_player_id', 'opening_id']].rename(
        columns={'black_player_id': 'player_id'}
    )
    samples = pd.concat([white, black], ignore_index=True)

    num_players = int(samples['player_id'].max()) + 1
    num_openings = int(samples['opening_id'].max()) + 1

    train_idx, val_idx = train_test_split(
        np.arange(len(samples)), test_size=VAL_SPLIT, random_state=42
    )
    train_df = samples.iloc[train_idx].reset_index(drop=True)
    val_df = samples.iloc[val_idx].reset_index(drop=True)

    print(f"Train positives: {len(train_df)}")
    print(f"Val positives: {len(val_df)}")
    print(f"Players: {num_players}, Openings: {num_openings}")

    return train_df, val_df, num_players, num_openings


def build_batch(player_b: torch.Tensor, opening_b: torch.Tensor, num_openings: int):
    """Concatenate positives with NEG_PER_POS random negatives per positive."""
    batch_size = player_b.size(0)
    neg_openings = torch.randint(
        0, num_openings, (batch_size, NEG_PER_POS), device=player_b.device
    )
    neg_players = player_b.unsqueeze(1).expand(-1, NEG_PER_POS).reshape(-1)

    all_players = torch.cat([player_b, neg_players])
    all_openings = torch.cat([opening_b, neg_openings.reshape(-1)])
    labels = torch.cat([
        torch.ones(batch_size, device=player_b.device),
        torch.zeros(batch_size * NEG_PER_POS, device=player_b.device),
    ])
    return all_players, all_openings, labels


def run_epoch(model, loader, num_openings, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for player_b, opening_b in loader:
            player_b = player_b.to(device)
            opening_b = opening_b.to(device)

            players, openings, labels = build_batch(player_b, opening_b, num_openings)
            logits = model(players, openings)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    train_df, val_df, num_players, num_openings = prepare_data('games_encoded.csv')

    train_loader = DataLoader(
        PositivesDataset(train_df['player_id'].values, train_df['opening_id'].values),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        PositivesDataset(val_df['player_id'].values, val_df['opening_id'].values),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MFModel(num_players, num_openings, EMBEDDING_DIM).to(device)
    print("\nModel architecture:")
    print(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"\nTraining for {EPOCHS} epochs (neg/pos = {NEG_PER_POS})")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, num_openings, criterion, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, num_openings, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_tf_style_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    model.load_state_dict(torch.load('best_tf_style_model.pth'))
    torch.save(model.state_dict(), 'tf_style_model.pth')
    np.save('player_embeddings.npy', model.player_emb.weight.detach().cpu().numpy())
    np.save('opening_embeddings.npy', model.opening_emb.weight.detach().cpu().numpy())
    print("Saved tf_style_model.pth, player_embeddings.npy, opening_embeddings.npy")


if __name__ == '__main__':
    main()
