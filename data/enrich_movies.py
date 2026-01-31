import argparse
import os
import re
import time
import pandas as pd
import requests
from tqdm import tqdm

TMDB_BASE_URL = "https://api.themoviedb.org/3"


def parse_title_year(title_str):
    match = re.match(r"^(.+)\s+\((\d{4})\)$", title_str.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title_str.strip(), None


def search_movie_tmdb(title, year, api_key):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": api_key,
        "query": title,
        "year": year,
        "language": "en-US",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("results"):
            return data["results"][0].get("overview", "")

        if year:
            params.pop("year")
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("results"):
                return data["results"][0].get("overview", "")

        return ""

    except requests.RequestException as e:
        print(f"Error fetching '{title}': {e}")
        return ""


def enrich_movies(movies_path, output_path, api_key=None, delay=0.25):
    if api_key is None:
        api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise ValueError("TMDB API key required")

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )

    print(f"Loaded {len(movies)} movies")

    parsed = movies["Title"].apply(parse_title_year)
    titles = [p[0] for p in parsed]
    years = [p[1] for p in parsed]

    # just a weird movie that we can't parse
    titles[988] = "L'associé"
    years[988] = 1982

    descriptions = []
    for title, year in tqdm(zip(titles, years), total=len(titles), desc="Fetching descriptions"):
        desc = search_movie_tmdb(title, year, api_key)
        descriptions.append(desc)
        time.sleep(delay)

    result = pd.DataFrame({
        "movie_id": movies["MovieID"],
        "title": titles,
        "year": pd.array(years, dtype="Int64"),
        "genres": movies["Genres"],
        "description": descriptions,
    })

    found = sum(1 for d in descriptions if d)
    print(f"Found descriptions for {found}/{len(result)} movies ({100 * found / len(result):.1f}%)")

    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich MovieLens movies with TMDB descriptions")
    parser.add_argument("--input", "-i", default="movies.dat", help="Input movies.dat path")
    parser.add_argument("--output", "-o", default="movies_enriched.csv", help="Output CSV path")
    parser.add_argument("--api-key", help="TMDB API key (or set TMDB_API_KEY env var)")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API calls")

    args = parser.parse_args()
    enrich_movies(args.input, args.output, args.api_key, args.delay)