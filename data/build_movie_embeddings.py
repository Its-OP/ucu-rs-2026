import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate movie embeddings with weighted field fusion"
    )
    parser.add_argument("input_csv", help="Path to movies CSV file")
    parser.add_argument("output_npz", help="Path for output .npz file")

    parser.add_argument("--col-title", default="title", help="Title column")
    parser.add_argument("--col-genres", default="genres", help="Genres column")
    parser.add_argument("--col-year", default="year", help="Year column")
    parser.add_argument("--col-description", default="description", help="Description column")
    parser.add_argument("--genre-delimiter", default="|", help="Delimiter for genres (default: |)")

    parser.add_argument("--weight-title", type=float, default=0.3)
    parser.add_argument("--weight-genres", type=float, default=0.2)
    parser.add_argument("--weight-year", type=float, default=0.1)
    parser.add_argument("--weight-description", type=float, default=0.4)

    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceBERT model (default: all-MiniLM-L6-v2, DistilBERT-based)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, mps, or None for auto")

    return parser.parse_args()


def prepare_field_text(df: pd.DataFrame, col: str) -> list[str]:
    """Convert a column to list of strings (required for embedding), handling missing values."""
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found, using empty strings")
        return [""] * len(df)

    texts = []
    for val in df[col]:
        if pd.isna(val) or val == "":
            texts.append("")
        else:
            texts.append(str(val))
    return texts


def encode_field(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts, returning zero vectors for empty strings."""
    embeddings = []
    dim = model.get_sentence_embedding_dimension()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[i:i + batch_size]
        batch_embeddings = []

        non_empty_indices = [idx for idx, text in enumerate(batch) if text.strip()]
        non_empty_texts = [batch[idx] for idx in non_empty_indices]

        encoded = model.encode(non_empty_texts, show_progress_bar=False) if non_empty_texts else []

        enc_idx = 0
        for j in range(len(batch)):
            if j in non_empty_indices:
                batch_embeddings.append(encoded[enc_idx])
                enc_idx += 1
            else:
                batch_embeddings.append(np.zeros(dim))

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def encode_genres(model: SentenceTransformer, genre_lists: list[list[str]], batch_size: int) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()

    # Collect all unique genres for batch encoding
    all_genres = set()
    for genres in genre_lists:
        all_genres.update(genres)
    all_genres.discard("")

    if not all_genres:
        return np.zeros((len(genre_lists), dim))

    # Encode and 'cache' all unique genres
    genre_list = list(all_genres)
    print(f"  Found {len(genre_list)} unique genres")
    genre_texts = [f"Movie belongs to {g} genre" for g in genre_list]
    genre_embeddings = model.encode(genre_texts, batch_size=batch_size, show_progress_bar=False)
    genre_to_embedding = {genre: genre_embeddings[idx] for idx, genre in enumerate(genre_list)}

    # Average genre embeddings per movie
    result = np.zeros((len(genre_lists), dim))
    for idx, genres in enumerate(genre_lists):
        valid_genres = [genre for genre in genres if genre in genre_to_embedding]
        if valid_genres:
            embs = np.array([genre_to_embedding[genre] for genre in valid_genres])
            result[idx] = embs.mean(axis=0)

    return result


def parse_genres(df: pd.DataFrame, col: str, delimiter: str = "|") -> list[list[str]]:
    if col not in df.columns:
        print(f"WARN: Column '{col}' not found")
        return [[] for _ in range(len(df))]

    result = []
    for val in df[col]:
        if pd.isna(val) or val == "":
            result.append([])
        else:
            genres = [g.strip() for g in str(val).split(delimiter) if g.strip()]
            result.append(genres)
    return result


def encode_years(model: SentenceTransformer, df: pd.DataFrame, col: str) -> np.ndarray:
    """Encode unique years once, look up per movie."""
    dim = model.get_sentence_embedding_dimension()

    if col not in df.columns:
        print(f"WARN: Column '{col}' not found")
        return np.zeros((len(df), dim))

    unique_years = set()
    for val in df[col]:
        if pd.notna(val) and val != "":
            try:
                unique_years.add(int(val))
            except (ValueError, TypeError):
                pass

    if not unique_years:
        return np.zeros((len(df), dim))

    # Encode all unique years once
    year_list = sorted(unique_years)
    print(f"  Found {len(year_list)} unique years ({min(year_list)}-{max(year_list)})")
    year_texts = [f"Released in {y}" for y in year_list]
    year_embeddings = model.encode(year_texts, show_progress_bar=False)
    year_to_emb = {y: year_embeddings[i] for i, y in enumerate(year_list)}

    # Look up per movie
    result = np.zeros((len(df), dim))
    for i, val in enumerate(df[col]):
        if pd.notna(val) and val != "":
            try:
                year = int(val)
                if year in year_to_emb:
                    result[i] = year_to_emb[year]
            except (ValueError, TypeError):
                pass

    return result


def weighted_fusion(embeddings: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine field embeddings via weighted average, handling missing fields."""
    n_samples = list(embeddings.values())[0].shape[0]
    dim = list(embeddings.values())[0].shape[1]
    result = np.zeros((n_samples, dim))

    for i in range(n_samples):
        total_weight = 0.0
        combined = np.zeros(dim)

        for field, emb_array in embeddings.items():
            vec = emb_array[i]
            if np.linalg.norm(vec) > 1e-9:  # non-zero embedding
                w = weights.get(field, 0.0)
                combined += w * vec
                total_weight += w

        if total_weight > 0:
            combined /= total_weight
            combined /= (np.linalg.norm(combined) + 1e-9)  # normalize

        result[i] = combined

    return result


def concatenate_fusion(embeddings: dict[str, np.ndarray], field_order: list[str]) -> np.ndarray:
    """Concatenate field embeddings in specified order."""
    arrays = [embeddings[f] for f in field_order if f in embeddings]
    result = np.concatenate(arrays, axis=1)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    return result / norms


def main():
    args = parse_args()

    weights = {
        "title": args.weight_title,
        "genres": args.weight_genres,
        "year": args.weight_year,
        "description": args.weight_description,
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        print(f"Note: Weights sum to {total:.2f} instead of 1")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)

    print(f"Reading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Found {len(df)} movies")

    # Field order for concatenation
    field_order = ["title", "genres", "year", "description"]

    embeddings = {}

    print(f"Encoding title (weight={weights['title']})...")
    titles = prepare_field_text(df, args.col_title)
    embeddings["title"] = encode_field(model, titles, args.batch_size)

    print(f"Encoding genres (weight={weights['genres']})...")
    genre_lists = parse_genres(df, args.col_genres, args.genre_delimiter)
    embeddings["genres"] = encode_genres(model, genre_lists, args.batch_size)

    print(f"Encoding year (weight={weights['year']})...")
    embeddings["year"] = encode_years(model, df, args.col_year)

    print(f"Encoding description (weight={weights['description']})...")
    descriptions = prepare_field_text(df, args.col_description)
    embeddings["description"] = encode_field(model, descriptions, args.batch_size)

    print("Creating weighted fusion")
    weighted_embeddings = weighted_fusion(embeddings, weights)

    print("Creating concatenated fusion")
    concat_embeddings = concatenate_fusion(embeddings, field_order)

    output_path = args.output_npz if args.output_npz.endswith(".npz") else f"{args.output_npz}.npz"
    print(f"Saving to: {output_path}")
    np.savez_compressed(
        output_path,
        title=embeddings["title"],
        genres=embeddings["genres"],
        year=embeddings["year"],
        description=embeddings["description"],
        weighted=weighted_embeddings,
        concat=concat_embeddings,
    )

    print(f"Done. Keys and shapes:")
    print(f"  title: {embeddings['title'].shape}")
    print(f"  genres: {embeddings['genres'].shape}")
    print(f"  year: {embeddings['year'].shape}")
    print(f"  description: {embeddings['description'].shape}")
    print(f"  weighted: {weighted_embeddings.shape}")
    print(f"  concat: {concat_embeddings.shape}")


if __name__ == "__main__":
    main()