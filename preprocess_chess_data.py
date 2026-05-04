"""Preprocess chess game data from CSV for machine learning.

This script:
1. Loads the extracted chess games CSV (with ratings)
2. Explores and displays data statistics
3. Encodes categorical features (player names, openings) to discrete integer values
4. Saves encoded data and encoders for later use
5. Appends to the encoded CSV when it already exists

Usage:
    python preprocess_chess_data.py --input games.csv --output games_encoded.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess and encode chess game data for ML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("games.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("games_encoded.csv"),
        help="Path to the output encoded CSV file.",
    )
    parser.add_argument(
        "--encoders",
        type=Path,
        default=Path("encoders.pkl"),
        help="Path to save the encoders for later use.",
    )
    return parser.parse_args()


def count_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def ensure_encoder(encoder: LabelEncoder | None) -> LabelEncoder:
    if encoder is None:
        encoder = LabelEncoder()
        encoder.classes_ = np.array([], dtype=object)
    return encoder


def encode_with_encoder(
    values: pd.Series, encoder: LabelEncoder
) -> tuple[pd.Series, LabelEncoder]:
    existing_classes = list(encoder.classes_)
    mapping = {cls: idx for idx, cls in enumerate(existing_classes)}

    new_values = [value for value in pd.unique(values) if value not in mapping]
    if new_values:
        new_values_sorted = sorted(new_values)
        encoder.classes_ = np.array(existing_classes + new_values_sorted, dtype=object)
        for value in new_values_sorted:
            mapping[value] = len(mapping)

    encoded = values.map(mapping)
    if encoded.isnull().any():
        missing = values[encoded.isnull()].unique()
        raise ValueError(f"Missing encoder mappings for values: {missing}")

    return encoded.astype(int), encoder


def explore_data(df: pd.DataFrame) -> None:
    """Display basic statistics about the dataset."""
    print("\n=== DATA EXPLORATION ===")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}\n")
    
    print("Data preview:")
    print(df.head(10))
    
    print(f"\nMissing values:\n{df.isnull().sum()}\n")
    
    print(f"Unique white players: {df['white_player'].nunique()}")
    print(f"Unique black players: {df['black_player'].nunique()}")
    print(f"Unique openings: {df['opening'].nunique()}")
    
    print(f"\nResult distribution:\n{df['result'].value_counts()}\n")
    print(f"Winning color distribution:\n{df['winning_color'].value_counts()}\n")


def encode_categorical_features(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categorical features to discrete integer values."""
    print("\n=== ENCODING CATEGORICAL FEATURES ===")
    
    encoders = encoders or {}
    df_encoded = df.copy()
    
    # Encode player names
    for col in ["white_player", "black_player", "winner", "loser"]:
        encoder = ensure_encoder(encoders.get(col))
        encoded, encoder = encode_with_encoder(df[col], encoder)
        df_encoded[col + "_encoded"] = encoded
        encoders[col] = encoder
        sample_mapping = dict(zip(encoder.classes_[:5], range(min(5, len(encoder.classes_)))))
        print(f"\nEncoded '{col}': {len(encoder.classes_)} unique values")
        print(f"  Sample: {sample_mapping}")
    
    # Encode opening names
    encoder = ensure_encoder(encoders.get("opening"))
    encoded, encoder = encode_with_encoder(df["opening"], encoder)
    df_encoded["opening_encoded"] = encoded
    encoders["opening"] = encoder
    sample_mapping = dict(zip(encoder.classes_[:5], range(min(5, len(encoder.classes_)))))
    print(f"\nEncoded 'opening': {len(encoder.classes_)} unique values")
    print(f"  Sample: {sample_mapping}")
    
    # Encode winning_color
    encoder = ensure_encoder(encoders.get("winning_color"))
    encoded, encoder = encode_with_encoder(df["winning_color"], encoder)
    df_encoded["winning_color_encoded"] = encoded
    encoders["winning_color"] = encoder
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"\nEncoded 'winning_color': {len(encoder.classes_)} unique values")
    print(f"  Mapping: {mapping}")
    
    # Encode result
    encoder = ensure_encoder(encoders.get("result"))
    encoded, encoder = encode_with_encoder(df["result"], encoder)
    df_encoded["result_encoded"] = encoded
    encoders["result"] = encoder
    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"\nEncoded 'result': {len(encoder.classes_)} unique values")
    print(f"  Mapping: {mapping}")
    
    return df_encoded, encoders


def select_ml_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Select and organize features for machine learning."""
    ml_df = pd.DataFrame({
        "white_player_id": df_encoded["white_player_encoded"],
        "black_player_id": df_encoded["black_player_encoded"],
        "opening_id": df_encoded["opening_encoded"],
        "white_rating": df_encoded["white_rating"],
        "black_rating": df_encoded["black_rating"],
        "winning_color": df_encoded["winning_color_encoded"],
        "result_code": df_encoded["result_encoded"],
        "winner_id": df_encoded["winner_encoded"],
        "loser_id": df_encoded["loser_encoded"],
    })
    return ml_df


def main() -> None:
    args = parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    append_mode = args.output.exists()
    existing_rows = 0
    if append_mode and not args.encoders.exists():
        raise FileNotFoundError(
            f"Encoders not found for append mode: {args.encoders}. "
            "Delete the output file to rebuild from scratch."
        )
    
    # Load CSV
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    if append_mode:
        existing_rows = count_data_rows(args.output)
        if existing_rows > len(df):
            raise ValueError(
                "Encoded dataset has more rows than the source CSV. "
                "Delete the encoded output to rebuild from scratch."
            )
        if existing_rows == len(df):
            print("No new rows to encode. Exiting.")
            return
        df = df.iloc[existing_rows:].reset_index(drop=True)
    
    # Convert ratings to numeric and fill missing with median
    df['white_rating'] = pd.to_numeric(df['white_rating'], errors='coerce')
    df['black_rating'] = pd.to_numeric(df['black_rating'], errors='coerce')
    
    median_white = df['white_rating'].median()
    median_black = df['black_rating'].median()
    
    df['white_rating'] = df['white_rating'].fillna(median_white).astype(int)
    df['black_rating'] = df['black_rating'].fillna(median_black).astype(int)
    
    # Explore data
    explore_data(df)
    
    # Encode features
    existing_encoders = None
    if append_mode:
        with args.encoders.open("rb") as f:
            existing_encoders = pickle.load(f)
    df_encoded, encoders = encode_categorical_features(df, existing_encoders)
    
    # Select ML features
    ml_df = select_ml_features(df_encoded)
    
    # Save encoded data
    save_mode = "a" if append_mode else "w"
    write_header = not append_mode or existing_rows == 0
    ml_df.to_csv(args.output, index=False, mode=save_mode, header=write_header)
    print(f"\n✓ Encoded data saved to {args.output}")
    
    # Save encoders
    with args.encoders.open("wb") as f:
        pickle.dump(encoders, f)
    print(f"✓ Encoders saved to {args.encoders}")
    
    print(f"\nFinal ML dataset shape: {ml_df.shape}")
    print(f"Final ML dataset preview:\n{ml_df.head(10)}")
    print(f"\nRating statistics:")
    print(f"  White rating - Mean: {ml_df['white_rating'].mean():.1f}, Std: {ml_df['white_rating'].std():.1f}")
    print(f"  Black rating - Mean: {ml_df['black_rating'].mean():.1f}, Std: {ml_df['black_rating'].std():.1f}")


if __name__ == "__main__":
    main()
