"""Preprocess chess game data from CSV for machine learning.

This script:
1. Loads the extracted chess games CSV
2. Explores and displays data statistics
3. Encodes categorical features (player names, openings) to discrete integer values
4. Saves encoded data and encoders for later use

Usage:
    python preprocess_chess_data.py --input games.csv --output games_encoded.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
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


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categorical features to discrete integer values."""
    print("\n=== ENCODING CATEGORICAL FEATURES ===")
    
    encoders = {}
    df_encoded = df.copy()
    
    # Encode player names
    for col in ["white_player", "black_player", "winner", "loser"]:
        encoder = LabelEncoder()
        df_encoded[col + "_encoded"] = encoder.fit_transform(df[col])
        encoders[col] = encoder
        print(f"\nEncoded '{col}': {len(encoder.classes_)} unique values")
        print(f"  Sample: {encoder.classes_[:5]} → {encoder.transform(encoder.classes_[:5])}")
    
    # Encode opening names
    encoder = LabelEncoder()
    df_encoded["opening_encoded"] = encoder.fit_transform(df["opening"])
    encoders["opening"] = encoder
    print(f"\nEncoded 'opening': {len(encoder.classes_)} unique values")
    print(f"  Sample: {encoder.classes_[:5]} → {encoder.transform(encoder.classes_[:5])}")
    
    # Encode winning_color
    encoder = LabelEncoder()
    df_encoded["winning_color_encoded"] = encoder.fit_transform(df["winning_color"])
    encoders["winning_color"] = encoder
    print(f"\nEncoded 'winning_color': {len(encoder.classes_)} unique values")
    print(f"  Mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    
    # Encode result
    encoder = LabelEncoder()
    df_encoded["result_encoded"] = encoder.fit_transform(df["result"])
    encoders["result"] = encoder
    print(f"\nEncoded 'result': {len(encoder.classes_)} unique values")
    print(f"  Mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    
    return df_encoded, encoders


def select_ml_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Select and organize features for machine learning."""
    ml_df = pd.DataFrame({
        "white_player_id": df_encoded["white_player_encoded"],
        "black_player_id": df_encoded["black_player_encoded"],
        "opening_id": df_encoded["opening_encoded"],
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
    
    # Load CSV
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Explore data
    explore_data(df)
    
    # Encode features
    df_encoded, encoders = encode_categorical_features(df)
    
    # Select ML features
    ml_df = select_ml_features(df_encoded)
    
    # Save encoded data
    ml_df.to_csv(args.output, index=False)
    print(f"\n✓ Encoded data saved to {args.output}")
    
    # Save encoders
    with args.encoders.open("wb") as f:
        pickle.dump(encoders, f)
    print(f"✓ Encoders saved to {args.encoders}")
    
    print(f"\nFinal ML dataset shape: {ml_df.shape}")
    print(f"Final ML dataset preview:\n{ml_df.head(10)}")


if __name__ == "__main__":
    main()
