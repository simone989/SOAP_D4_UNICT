import os
import json
import logging
import time
import numpy as np
import spacy
import uuid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt  # Per il plotting

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_law_text(filepath: str) -> str:
    """
    Carica il contenuto del file di testo.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def extract_nouns(text: str, nlp) -> list:
    """
    Estrae tutti i sostantivi dal testo utilizzando spaCy.
    """
    doc = nlp(text)
    nouns = []
    for sentence in doc.sents:
        for token in sentence:
            if token.pos_ == "NOUN":
                nouns.append(token.text)
    return nouns

def get_noun_vectors(nouns: list, nlp) -> np.ndarray:
    """
    Ottiene il vettore associato ad ogni sostantivo.
    """
    vectors = [nlp.vocab.get_vector(word) for word in nouns]
    return np.array(vectors)

def plot_silhouette_scores(vectors: np.ndarray, min_k: int = 2, max_k: int = 300) -> (int, float, list, list):
    """
    Calcola e logga i silhouette score per k nel range [min_k, max_k],
    restituendo il k ottimale, il relativo score e le liste dei valori calcolati.
    """
    k_values = []
    silhouette_scores = []
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42)
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        k_values.append(k)
        silhouette_scores.append(score)
        logging.info(f"Silhouette score for k={k}: {score}")
        if score > best_score:
            best_score = score
            best_k = k
    # Se vuoi visualizzare il plot, decommenta le righe sottostanti
    # plt.figure(figsize=(10,6))
    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Scores for different values of k')
    # plt.grid(True)
    # plt.show()
    return best_k, best_score, k_values, silhouette_scores

def extract_k_by_threshold(k_values: list, silhouette_scores: list, threshold: float) -> int:
    """
    Restituisce il più piccolo k per cui il silhouette score è >= threshold.
    Se nessun k raggiunge la soglia, restituisce None.
    """
    candidate_ks = [k for k, score in zip(k_values, silhouette_scores) if score >= threshold]
    if candidate_ks:
        return min(candidate_ks)
    return None

def cluster_nouns(vectors: np.ndarray, n_clusters: int) -> list:
    """
    Applica KMeans sui vettori dei sostantivi.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42)
    kmeans.fit(vectors)
    return kmeans.labels_

def build_clustered_nouns(nouns: list, labels: list) -> dict:
    """
    Associa ogni sostantivo al proprio cluster.
    """
    clustered = {}
    for label, noun in zip(labels, nouns):
        if label not in clustered:
            clustered[label] = []
        clustered[label].append(noun.upper())
    return clustered

def filter_clusters_by_number_of_elements(clustered_nouns: dict, min_elements: int) -> dict:
    """
    Filtra i cluster mantenendo solo quelli con più di min_elements elementi.
    """
    return {cluster: words for cluster, words in clustered_nouns.items() if len(words) > min_elements}

def average_pairwise_similarity(vectors: np.ndarray) -> float:
    """
    Calcola la similarità coseno media tra tutte le coppie di vettori.
    Se il cluster contiene un solo elemento, restituisce 1.0.
    """
    if len(vectors) < 2:
        return 1.0

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-10, norms)
    norm_vectors = vectors / safe_norms

    sim_matrix = np.dot(norm_vectors, norm_vectors.T)
    n = sim_matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_indices]

    return np.mean(similarities)

def filter_clusters(clustered_nouns: dict, vectors: np.ndarray, labels: list, similarity_threshold: float) -> dict:
    """
    Filtra i cluster mantenendo quelli con similarità media >= similarity_threshold.
    """
    filtered = {}
    for cluster, words in clustered_nouns.items():
        indices = [i for i, label in enumerate(labels) if label == cluster]
        cluster_vectors = vectors[indices]
        avg_sim = average_pairwise_similarity(cluster_vectors)
        if avg_sim >= similarity_threshold:
            filtered[cluster] = words
    return filtered

def compute_centroids(vectors: np.ndarray, labels: list) -> dict:
    """
    Calcola il centroide di ciascun cluster.
    """
    centroids = {}
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        indices = [i for i, label in enumerate(labels) if label == cluster]
        cluster_vectors = vectors[indices]
        centroids[cluster] = np.mean(cluster_vectors, axis=0)
    return centroids

def merge_similar_clusters(vectors: np.ndarray, labels: list, merge_similarity_threshold: float) -> dict:
    """
    Unisce i cluster i cui centroidi sono semanticamente simili.
    """
    centroids_dict = compute_centroids(vectors, labels)
    sorted_clusters = sorted(centroids_dict.keys())
    centroids = np.array([centroids_dict[c] for c in sorted_clusters])
    pairwise_distances = pdist(centroids, metric="cosine")
    Z = linkage(pairwise_distances, method="average")
    distance_threshold = 1 - merge_similarity_threshold
    new_labels = fcluster(Z, t=distance_threshold, criterion="distance")
    mapping = {}
    for orig_label, new_label in zip(sorted_clusters, new_labels):
        mapping.setdefault(new_label, []).append(orig_label)
    return mapping

def merge_clusters(vectors: np.ndarray, labels: list, filtered_clusters: dict, merge_similarity_threshold: float) -> dict:
    """
    Unisce i cluster filtrati sulla base della similarità dei centroidi.
    """
    merge_mapping = merge_similar_clusters(vectors, labels, merge_similarity_threshold)
    merged_clusters = {}
    for new_cluster, orig_clusters in merge_mapping.items():
        merged_words = []
        for orig in orig_clusters:
            if orig in filtered_clusters:
                merged_words.extend(filtered_clusters[orig])
        if merged_words:
            merged_clusters[new_cluster] = merged_words
    return merged_clusters

def remove_duplicates(word_list, nlp):
    """
    Rimuove duplicati preservando l'ordine originale, usando il lemma di ogni parola.
    """
    seen = set()
    unique_words = []
    for word in word_list:
        token = nlp(word)[0]
        lemma = token.lemma_.lower()
        if lemma not in seen:
            seen.add(lemma)
            unique_words.append(word)
    return unique_words



import uuid
from datetime import datetime

def prepare_messages(merged_clusters: dict) -> list:
    """
    Prepara una lista di messaggi per la richiesta batch, separando le istruzioni e i dati.

    Il primo messaggio ha ruolo "assistant" e contiene le istruzioni, arricchite con elementi dinamici
    (come la data di oggi) per evitare il caching del prompt, e include anche un esempio di output atteso.
    Il secondo messaggio ha ruolo "user" e contiene la rappresentazione testuale dei cluster.

    A ciascun messaggio viene aggiunto un hash casuale (32 caratteri) alla fine.
    """
    today = datetime.today().strftime("%Y-%m-%d")

    instructions = (
        f"You are an expert in cybersecurity, digital forensics, and threat assessment.\n\n"
        f"As of {today}, I have extracted clusters of nouns from a legal text as part of a process aimed at threat elicitation. "
        "Threat elicitation is the process of identifying potential threats and vulnerabilities in order to assess risks and plan effective countermeasures. "
        "In this context, the assets to be identified are those concepts, resources, or areas that the law intends to protect and that are functional "
        "for supporting the threat elicitation process.\n\n"
        f"This request was generated on {today} and reflects the latest analysis of the data.\n\n"
        "Your task is to analyze each cluster and determine which clusters represent assets. You may select more than one cluster; "
        "choose all clusters that you believe contain assets, even if only partially. For each cluster that you identify as an asset, "
        "output its identifier and the list of terms it contains. Please do not include any commentary or explanation—simply return "
        "the selected clusters and their content in JSON format (keys being the cluster IDs and values being the lists of terms).\n\n"
        "Output only the result as valid JSON. Do not include any additional text, markdown formatting, or explanation.\n\n"
        f"Ensure that your response is accurate as of {today}.\n\n"
        "For example, your output should be in the following format:\n\n"
        "{\n"
        '  "1": [ "X", "Y", "Z" ],\n'
        '  "2": [ "X", "Y", "Z" ],\n'
        '  "6": [ "X", "Y", "Z" ]\n'
        "}"
    )

    clusters_str = "\n".join([f"Cluster {cluster_id}: {words}" for cluster_id, words in merged_clusters.items()])


    messages = [
        {"role": "assistant", "content": f"{instructions}"},
        {"role": "user", "content": f"{clusters_str}"}
    ]
    return messages


def deprecated_call_ai(prompt: str) -> dict:
    """
    [DEPRECATED] Invia il prompt al modello AI e restituisce la response in formato JSON.
    Questa funzione verrà mantenuta per eventuali usi futuri.
    """
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    message = [{"role": "assistant", "content": prompt}]
    response = client.chat.completions.create(
        model="o1-mini",
        messages=message,
    )
    text_response = response.choices[0].message.content
    text_response = text_response.replace("```json\n", "").replace("```", "")
    try:
        result_json = json.loads(text_response)
    except Exception as e:
        logging.error(f"Errore nel parsing della response: {e}")
        result_json = {}
    return result_json

def call_ai_batch(batch_entries: list, batch_filename: str = None) -> str:
    """
    Crea un file batch JSONL contenente le richieste degli esperimenti e lo invia tramite il batch endpoint di OpenAI.
    Restituisce il batch ID.
    """
    if batch_filename is None:
        # Utilizza un timestamp per rendere unico il nome del file
        batch_filename = f"batch_experiments_{int(time.time())}.jsonl"

    # Scrive le richieste in un file JSONL
    with open(batch_filename, "w", encoding="utf-8") as batch_file:
        for entry in batch_entries:
            batch_file.write(json.dumps(entry) + "\n")
    logging.info(f"Batch file '{batch_filename}' creato con {len(batch_entries)} richieste.")

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Carica il file batch su OpenAI
    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    logging.info(f"File batch caricato con ID: {batch_input_file_id}")

    # Crea il batch per il file
    batch_response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch processing for experiments from file {batch_filename} with {len(batch_entries)} requests."
        }
    )
    batch_id = batch_response.id
    logging.info(f"Batch creato con ID: {batch_id}")
    return batch_id

def remove_duplicates_from_clusters(result_json: dict, nlp) -> dict:
    """
    Rimuove i duplicati da ciascun cluster basandosi sul lemma di ogni parola.
    """
    final_clusters = {}
    for cluster_id, words in result_json.items():
        final_clusters[cluster_id] = remove_duplicates(words, nlp)
    return final_clusters

def run_experiments(vectors: np.ndarray, nouns: list, nlp, min_k: int = 2, max_k: int = 50):
    """
    Esegue una serie di esperimenti variando i parametri:
      - threshold per la selezione del numero di cluster
      - similarity_threshold per il filtraggio dei cluster
      - merge_similarity_threshold per il merge dei cluster
    Per ogni combinazione, viene preparato il prompt per l'AI.
    Al termine il tutto viene inviato in un'unica richiesta batch.
    I risultati intermedi vengono loggati e salvati in un file JSON.
    """
    logging.info("Inizio calcolo silhouette score...")
    optimal_k, best_score, k_values, silhouette_scores = plot_silhouette_scores(vectors, min_k, max_k)
    logging.info(f"Silhouette score calcolati: best_k={optimal_k}, best_score={best_score}")

    # Range dei parametri da testare
    threshold_values = [0.1, 0.15, 0.2, 0.25, 0.30]
    similarity_threshold_values = [0.5, 0.6, 0.7]
    merge_similarity_threshold_values = [0.7, 0.75, 0.8]

    experiments_results = []
    batch_entries = []
    experiment_count = 0

    for threshold in threshold_values:
        for similarity_threshold in similarity_threshold_values:
            for merge_similarity_threshold in merge_similarity_threshold_values:
                experiment_count += 1
                logging.info(f"Inizio esperimento {experiment_count}: threshold={threshold}, "
                             f"similarity_threshold={similarity_threshold}, "
                             f"merge_similarity_threshold={merge_similarity_threshold}")

                selected_k = extract_k_by_threshold(k_values, silhouette_scores, threshold)
                if selected_k is None:
                    selected_k = optimal_k
                    logging.info(f"Nessun k soddisfa il threshold {threshold}, uso optimal_k={optimal_k}")
                else:
                    logging.info(f"Selected k in base al threshold {threshold}: {selected_k}")

                labels = cluster_nouns(vectors, selected_k)
                clustered_nouns = build_clustered_nouns(nouns, labels)
                filtered_clusters = filter_clusters(clustered_nouns, vectors, labels, similarity_threshold)
                merged_clusters = merge_clusters(vectors, labels, filtered_clusters, merge_similarity_threshold)

                logging.info("Prompt preparato per esperimento {0}.".format(experiment_count))

                # Invece di chiamare l'AI singolarmente, creiamo un batch entry
                custom_id = (f"batch-exp-{experiment_count}-th{threshold}-sim{similarity_threshold}-"
                             f"merge{merge_similarity_threshold}-{int(time.time())}")
                entry = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "o1-mini",
                        "messages": prepare_messages(merged_clusters)
                    }
                }
                batch_entries.append(entry)

                # Salva anche i risultati intermedi per eventuali analisi
                experiment_result = {
                    "threshold": threshold,
                    "similarity_threshold": similarity_threshold,
                    "merge_similarity_threshold": merge_similarity_threshold,
                    "custom_id": custom_id,
                    #"selected_k": selected_k,
                    #"merged_clusters": merged_clusters
                }
                experiments_results.append(experiment_result)
                logging.info(f"Esperimento {experiment_count} completato.")

    # Salva i risultati intermedi in un file JSON
    with open("experiment_intermediate_results.json", "w", encoding="utf-8") as f:
        json.dump(experiments_results, f, indent=2)
    logging.info("Risultati intermedi salvati in 'experiment_intermediate_results.json'.")

    # Invia in batch tutte le richieste
    batch_id = call_ai_batch(batch_entries)
    logging.info(f"Tutti gli esperimenti sono stati inviati in batch con ID: {batch_id}")

def main():
    filepath = "" ## Documents path
    logging.info(f"Caricamento del file: {filepath}")
    law_text = load_law_text(filepath)

    nlp = spacy.load("en_core_web_lg")
    logging.info("Modello spaCy caricato.")
    nouns = extract_nouns(law_text, nlp)
    vectors = get_noun_vectors(nouns, nlp)

    logging.info("Inizio esecuzione esperimenti...")
    run_experiments(vectors, nouns, nlp, min_k=2, max_k=50)

if __name__ == "__main__":
    main()
