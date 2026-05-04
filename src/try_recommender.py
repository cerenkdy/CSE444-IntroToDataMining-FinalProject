"""
Try the journal recommender from the command line.

Usage:
    python src/try_recommender.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from journal_finder import load_data, JournalFinder

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "CompSciencePub.sqlite")


def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        print("Place CompSciencePub.sqlite in the data/ folder and try again.")
        return

    print("Loading the corpus and training the recommender...")
    print("(this takes 1-2 minutes on the first run)\n")

    df = load_data(DB_PATH)
    print(f"  loaded {len(df):,} articles across {df['journal'].nunique()} journals")

    finder = JournalFinder().fit(df)
    print("  ready.\n")

    print("=" * 60)
    print("Journal Finder - paste an abstract, get the top 5 journals")
    print("=" * 60)
    print("Type or paste your abstract, then press Enter twice (empty line) to submit.")
    print("Type 'quit' on a line by itself to exit.\n")

    while True:
        print("Abstract:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() == "quit":
                return
            if line == "" and lines:
                break
            if line:
                lines.append(line)

        abstract = " ".join(lines).strip()
        if not abstract:
            continue

        print("\nTop 5 recommended journals:")
        print("-" * 60)
        for rank, (journal, score) in enumerate(finder.top_5(abstract), start=1):
            print(f"  {rank}. {journal}  (score: {score:.4f})")
        print()


if __name__ == "__main__":
    main()
