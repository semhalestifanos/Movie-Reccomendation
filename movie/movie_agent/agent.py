import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from google.adk.agents import Agent
GEMINI_MODEL = "gemini-2.5-flash"

# Build a robust path to the data file relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "data", "imdb_top_1000.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_movie_db"
COLLECTION_NAME = "movies"

# globals
embedding_model = None
chroma_client = None
chroma_collection = None
movies_df = None

def load_data() -> str:
    """Load IMDB dataset into memory and ChromaDB collection."""
    global movies_df, embedding_model, chroma_client, chroma_collection

    if movies_df is not None:
        return "Data already loaded."

    movies_df = pd.read_csv(CSV_PATH)[["Series_Title", "Overview"]]
    movies_df.columns = ["title", "description"]

    embedding_model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if chroma_collection.count() == 0:
        ids = [str(i) for i in range(len(movies_df))]
        texts = (movies_df["title"] + ". " + movies_df["description"]).tolist()
        embeddings = embedding_model.encode(texts, convert_to_numpy=True).tolist()
        metas = [{"title": t, "description": d} for t, d in zip(movies_df["title"], movies_df["description"])]
        chroma_collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    return f"Loaded {len(movies_df)} movies."

def recommend_movies(query: str, top_k: int = 5) -> list:
    """Return top movie recommendations for a query."""
    global embedding_model, chroma_collection
    if embedding_model is None or chroma_collection is None:
        load_data()

    q_vec = embedding_model.encode([query], convert_to_numpy=True).tolist()
    results = chroma_collection.query(query_embeddings=q_vec, n_results=top_k, include=["metadatas", "distances"])
    recs = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        recs.append({
            "title": meta["title"],
            "description": meta["description"],
            "score": 1 - float(dist)
        })
    return recs

import random

def explain_choice(movie_title: str, query: str) -> str:
    """Return a witty randomized explanation if Gemini is unavailable."""
    templates = [
        f"🎬 '{movie_title}' is a spot-on pick for your vibe '{query}'. "
        f"It’s like the universe heard your mood and wrapped it into a film!",
        
        f"🍿 '{movie_title}' nails your '{query}' mood — basically your personality in HD!",
        
        f"✨ If '{query}' were a movie feeling, '{movie_title}' would be the director’s cut just for you.",
        
        f"😎 '{movie_title}' matches your '{query}' vibe perfectly — it’s like it crawled out of your watchlist dreams.",
        
        f"🎥 Looking for '{query}' energy? '{movie_title}' delivers it with style, sass, and a great soundtrack."
    ]
    return random.choice(templates)


# Agent
root_agent = Agent(
    name="movie_recommender_agent",
    model=GEMINI_MODEL,
    description="Movie recommender that can search for movies and explain the recommendations.",
    instruction = """
You are a witty and knowledgeable movie recommender agent.

Your job:
1. Always begin by ensuring the movie dataset is loaded using `load_data`.
2. Take the user’s query (genre, mood, or style) and call `recommend_movies` to generate a shortlist of options.
3. For each recommendation, call `explain_choice` to give a fun, engaging reason why it fits the user’s request. 
   • Be witty and give more detail.
   • Use humor, cultural references, or playful commentary where appropriate.
4. Return the final response in a clear, user-friendly format (not raw JSON).

Your goal is to make the user feel like they’re getting recommendations from a clever movie buddy, 
not a boring catalog.
""",
    tools=[load_data, recommend_movies, explain_choice],
)
