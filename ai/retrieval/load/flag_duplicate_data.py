import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from collections import deque
import uuid

def flag_duplicate_sentences_from_qdrant(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "customer_support_tweets",
    threshold: float = 0.8,
    show_progress: bool = True,
    qdrant_scroll_limit: int = 10000,
    group_limit: int = 100
) -> pd.DataFrame:
    """
    Retrieves sentences and their embeddings from an existing Qdrant collection
    and flags semantically similar sentences by assigning them to groups.
    Also returns the similarity scores between grouped points.

    Returns:
        pd.DataFrame: A DataFrame with 'qdrant_id', 'input_sentence', 'reply', 'group', and 'grouped_with' columns.
                      'grouped_with' is a list of (neighbor_id, similarity_score) for each point in the same group.
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    if show_progress:
        print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")

    # Verify collection exists
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        if show_progress:
            print(f"Collection '{collection_name}' found. Total points: {collection_info.points_count}")
        total_points_count = collection_info.points_count
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found or inaccessible. {e}")
        return pd.DataFrame()

    # Retrieve all points (IDs, vectors, and payloads) from Qdrant
    all_points_data = []
    last_id = None
    if show_progress:
        print(f"Retrieving all {total_points_count} points from Qdrant...")

    with tqdm(total=total_points_count, desc="Retrieving points", disable=not show_progress) as pbar_retrieval:
        while True:
            scroll_result, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=qdrant_scroll_limit,
                offset=last_id,
                with_payload=True,
                with_vectors=True,
            )
            all_points_data.extend(scroll_result)
            pbar_retrieval.update(len(scroll_result))
            if next_page_offset is None:
                break
            last_id = next_page_offset

    if not all_points_data:
        print("No points found in the collection.")
        return pd.DataFrame()

    qdrant_id_to_local_idx = {point.id: i for i, point in enumerate(all_points_data)}
    n = len(all_points_data)
    group_labels = [None] * n
    group_counter = 1

    # For each point, store a list of (neighbor_id, similarity_score) for those grouped with it
    grouped_with_scores = [[] for _ in range(n)]

    # Only seed grouping from the first `group_limit` points, but allow BFS to expand to all points
    seed_indices = set(range(min(group_limit, n)))
    assigned_indices = set()

    if show_progress:
        print(f"Starting grouping process with Qdrant search (seeding from first {group_limit} points, can expand to all)...")

    pbar_grouping = tqdm(total=len(seed_indices), desc="Flagging duplicate sentences (seeds)", disable=not show_progress)

    # Warn if threshold is not set or is too low
    if threshold is None or threshold <= 0.0:
        print(
            "WARNING: The similarity threshold is not set or is too low (<= 0.0). "
            "This will cause all points to be grouped together, regardless of their actual similarity. "
            "Please set a reasonable threshold (e.g., 0.7 or 0.8) to avoid meaningless groupings."
        )

    for current_local_idx_seed in seed_indices:
        if group_labels[current_local_idx_seed] is not None:
            pbar_grouping.update(1)
            continue

        group_labels[current_local_idx_seed] = group_counter
        assigned_indices.add(current_local_idx_seed)
        pbar_grouping.update(1)

        q = deque([current_local_idx_seed])

        while q:
            current_bfs_local_idx = q.popleft()
            current_vector = all_points_data[current_bfs_local_idx].vector

            # Qdrant returns hits with .score attribute (cosine similarity)
            search_hits = client.search(
                collection_name=collection_name,
                query_vector=current_vector,
                query_filter=None,
                limit=min(n, 1000),
                score_threshold=threshold,
                with_payload=False,
                with_vectors=False
            )

            for hit in search_hits:
                neighbor_qdrant_id = hit.id
                neighbor_local_idx = qdrant_id_to_local_idx.get(neighbor_qdrant_id)
                if neighbor_local_idx is not None and group_labels[neighbor_local_idx] is None:
                    group_labels[neighbor_local_idx] = group_counter
                    assigned_indices.add(neighbor_local_idx)
                    q.append(neighbor_local_idx)
                    # Store the similarity score for this grouping
                    grouped_with_scores[current_bfs_local_idx].append((neighbor_qdrant_id, hit.score))
                    grouped_with_scores[neighbor_local_idx].append((all_points_data[current_bfs_local_idx].id, hit.score))
                elif neighbor_local_idx is not None and group_labels[neighbor_local_idx] == group_counter:
                    # Already grouped, but may not have score recorded (for symmetry)
                    grouped_with_scores[current_bfs_local_idx].append((neighbor_qdrant_id, hit.score))

        group_counter += 1

    pbar_grouping.close()

    # Convert grouped_with_scores to a more readable format (list of dicts)
    grouped_with_str = []
    for i, neighbors in enumerate(grouped_with_scores):
        # Remove duplicates and self
        unique_neighbors = {}
        for nid, score in neighbors:
            if nid != all_points_data[i].id:
                if nid not in unique_neighbors or score > unique_neighbors[nid]:
                    unique_neighbors[nid] = score
        grouped_with_str.append(
            [{"neighbor_id": nid, "score": score} for nid, score in unique_neighbors.items()]
        )

    results_df = pd.DataFrame({
        'qdrant_id': [p.id for p in all_points_data],
        'input_sentence': [p.payload.get('input') for p in all_points_data],
        'reply': [p.payload.get('reply') for p in all_points_data],
        'group': group_labels,
        'grouped_with': grouped_with_str
    })

    if results_df['input_sentence'].isnull().all():
        print("Warning: 'input' key not found or all values are null in Qdrant payload. Ensure your payload structure is correct.")

    return results_df

if __name__ == '__main__':
    group_limit = 1000000
    print("Attempting to flag duplicate sentences from existing Qdrant data (test mode, seeding from first 100)...")
    flagged_df = flag_duplicate_sentences_from_qdrant(
        collection_name="customer_support_tweets",
        threshold=0.95,
        qdrant_scroll_limit=10000,
        group_limit=group_limit
    )

    if not flagged_df.empty:
        print("\nFlagging complete!")
        print("Sample of flagged DataFrame (with grouped similarity scores):")
        print(flagged_df)
        flagged_df.to_csv(f"flagged_df_{group_limit}.csv", index=False)
