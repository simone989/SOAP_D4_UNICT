#!/usr/bin/env python
import os
import json
import logging
import argparse
import spacy
from dotenv import load_dotenv
from openai import OpenAI

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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


def remove_duplicates_from_clusters(result_json: dict, nlp) -> dict:
    """
    Rimuove i duplicati da ciascun cluster basandosi sul lemma di ogni parola.
    """
    final_clusters = {}
    for cluster_id, words in result_json.items():
        final_clusters[cluster_id] = remove_duplicates(words, nlp)
    return final_clusters

def retrieve_batch_results(batch_id: str, client: OpenAI) -> str:
    """
    Recupera il contenuto del file di risultati associato al batch_id.
    Supponiamo che il batch dettagliato contenga il campo 'output_file_id'.
    """
    batch_details = client.batches.retrieve(batch_id)
    result_file_id = batch_details.output_file_id
    content = client.files.content(result_file_id)
    text_content = content.text
    output_file = f"raw_batch_results_{batch_id}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_content)
    return text_content

def update_experiment_intermediate_results(final_results: dict, intermediate_file: str = "experiment_intermediate_results.json", batch_id: str = "") -> str:
    """
    Carica il file intermedio (experiment_intermediate_results.json) e per ogni entry,
    se il custom_id corrisponde a uno presente in final_results, aggiunge il campo 'final_clusters'.
    Salva poi il file aggiornato con un nuovo nome.
    """
    try:
        print(intermediate_file)
        with open(intermediate_file, "r", encoding="utf-8") as f:
            experiments = json.load(f)
    except Exception as e:
        logging.error(f"Errore nel caricamento di {intermediate_file}: {e}")
        experiments = []

    for exp in experiments:
        cid = exp.get("custom_id")
        if cid in final_results:
            exp["final_clusters"] = final_results[cid]

    updated_file = f"final_result_{batch_id}.json"
    with open(updated_file, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    logging.info(f"File intermedio aggiornato salvato in '{updated_file}'.")
    return updated_file

def process_batch_results(batch_id: str, nlp, intermediate_file: str = "experiment_intermediate_results.json"):
    """
    Recupera i risultati del batch a partire dal batch_id,
    elabora ciascuna riga del file dei risultati (JSONL) parsando la risposta,
    rimuovendo duplicati e salvando il risultato finale in un file.

    Inoltre, aggiorna il file intermedio (experiment_intermediate_results.json) inserendo
    il campo "final_clusters" per ciascuna entry identificata dal custom_id.
    """
    # # load_dotenv()
    # client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # logging.info(f"Recupero risultati per batch_id: {batch_id}")
    # try:
    #     content = retrieve_batch_results(batch_id, client)
    # except Exception as e:
    #     logging.error(f"Errore nel recupero dei risultati per batch {batch_id}: {e}")
    #     return

    input_file = f"raw_batch_results_{batch_id}.jsonl"
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        logging.info(f"File '{input_file}' letto con successo.")
    except Exception as e:
        logging.error(f"Errore nell'apertura del file '{input_file}': {e}")
        return

    final_results = {}
    # Il contenuto è in formato JSONL, una riga per richiesta
    for line in content.splitlines():
        try:
            result_entry = json.loads(line)

            custom_id = result_entry.get("custom_id")
            response = result_entry.get("response", {})
            text_response = response["body"]["choices"][0]["message"]["content"]
            text_response = text_response.replace("```json\n", "").replace("```", "")

            try:
                result_json = json.loads(text_response)
            except Exception as e:
                logging.error(f"Errore nel parsing della risposta JSON per custom_id {custom_id}: {e}")
                result_json = {}
            final_clusters = remove_duplicates_from_clusters(result_json, nlp)
            final_results[custom_id] = final_clusters
        except Exception as e:
            logging.error(f"Errore nell'elaborazione di una riga del batch {custom_id}: {e}")

    # Salva i risultati finali in un file separato
    output_file = f"batch_final_results_{batch_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Batch processing completato e risultati salvati in '{output_file}'.")

    # Aggiorna il file intermedio con i final results usando il custom_id
   #print(final_results)
    update_experiment_intermediate_results(final_results, intermediate_file, batch_id)

def main():
    parser = argparse.ArgumentParser(description="Processa i risultati del batch a partire dal batch_id e aggiorna il file intermedio.")
    parser.add_argument("batch_id", help="ID del batch da processare")
    parser.add_argument("--intermediate_file", default="experiment_intermediate_results.json", help="File intermedio da aggiornare")
    args = parser.parse_args()

    load_dotenv()
    nlp = spacy.load("en_core_web_lg")
    process_batch_results(args.batch_id, nlp, args.intermediate_file)

if __name__ == "__main__":
    main()
