"""Extract chess game metadata from a PGN file.

This script reads a PGN export from Lichess and writes one row per game with:
- player names
- ratings (ELO)
- opening name
- game result
- winner and loser names derived from the result

Usage:
    python extract_lichess_openings.py --input lichess.pgn --output games.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess.pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract player names, opening, and winner/loser info from a PGN file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("lichess.pgn"),
        help="Path to the source PGN file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("games.csv"),
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


def winner_and_loser(white: str, black: str, result: str) -> tuple[str, str, str]:
    if result == "1-0":
        return white, black, "White"
    if result == "0-1":
        return black, white, "Black"
    return "Draw", "Draw", "Draw"


def extract_games(pgn_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with pgn_path.open("r", encoding="utf-8", errors="replace") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            white_elo = game.headers.get("WhiteElo", "")
            black_elo = game.headers.get("BlackElo", "")
            opening = game.headers.get("Opening", "")
            result = game.headers.get("Result", "")

            winner, loser, winning_color = winner_and_loser(
                white_player,
                black_player,
                result,
            )

            rows.append(
                {
                    "white_player": white_player,
                    "black_player": black_player,
                    "white_rating": white_elo,
                    "black_rating": black_elo,
                    "opening": opening,
                    "result": result,
                    "winning_color": winning_color,
                    "winner": winner,
                    "loser": loser,
                }
            )

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "white_player",
        "black_player",
        "white_rating",
        "black_rating",
        "opening",
        "result",
        "winning_color",
        "winner",
        "loser",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = extract_games(args.input)
    write_csv(rows, args.output)
    print(f"Extracted {len(rows)} games to {args.output}")


if __name__ == "__main__":
    main()