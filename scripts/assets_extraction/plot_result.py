import json
import sys
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main(file_id):
    # Costruisci il nome del file in base all'ID passato in input
    filename = f"final_result_{file_id}.json"

    # Carica i dati dal file JSON
    with open(filename, 'r') as f:
        data = json.load(f)

    # Se il JSON contiene un singolo esperimento, lo trasformiamo in lista
    if isinstance(data, dict):
        experiments = [data]
    else:
        experiments = data

    # Carica il modello spaCy (assicurati di aver scaricato 'en_core_web_lg')
    try:
        nlp = spacy.load("en_core_web_lg")
    except Exception as e:
        print("Errore nel caricamento del modello spaCy. Assicurati di aver scaricato 'en_core_web_lg'.")
        sys.exit(1)

    # Liste per salvare i risultati per ogni esperimento
    experiment_ids = []
    asset_counts = []
    cohesion_values = []

    # Lista per raccogliere informazioni per la selezione dei top 3
    experiments_info = []

    for exp in experiments:
        exp_id = exp.get("custom_id", "Unknown")
        clusters = exp.get("final_clusters", {})

        # Unisci tutti gli asset provenienti da ogni cluster
        merged_assets = []
        for cluster_assets in clusters.values():
            merged_assets.extend(cluster_assets)

        # Salva merged_assets nel dizionario dell'esperimento
        exp['merged_assets'] = merged_assets

        num_assets = len(merged_assets)
        experiment_ids.append(exp_id)
        asset_counts.append(num_assets)

        # Calcola la coesione: se ci sono asset, calcola il vettore di ciascun asset tramite spaCy
        if num_assets > 0:
            vectors = [nlp(asset).vector for asset in merged_assets]
            X = np.array(vectors)
            # Applichiamo KMeans con un unico cluster per ottenere il centroide e l'inertia
            kmeans = KMeans(n_clusters=1, random_state=42).fit(X)
            inertia = kmeans.inertia_
            # Calcoliamo la coesione come inertia media (inertia diviso numero di asset)
            cohesion = inertia / num_assets
        else:
            cohesion = float('nan')

        cohesion_values.append(cohesion)
        experiments_info.append((exp_id, num_assets, cohesion))
        print(f"Esperimento: {exp_id}, Numero asset: {num_assets}, Coesione: {cohesion:.4f}" if num_assets > 0 else f"Esperimento: {exp_id}, Nessun asset.")

    # Salva il file JSON aggiornato
    with open(filename, 'w') as f:
        json.dump(experiments, f, indent=4)



    # --- Estrazione dei Top 3 esperimenti ---
    # Ordina gli esperimenti: ordiniamo in modo decrescente prima per numero di asset e poi per coesione.
    # Se invece la coesione migliore fosse indicata da un valore minore (maggiore compattezza), sostituisci 'cohesion'
    # con '-cohesion' nel lambda.
    top_3 = sorted(experiments_info, key=lambda x: (x[1], x[2]), reverse=True)[:3]

    print("\nTop 3 esperimenti (cluster) selezionati:")
    for idx, (exp_id, num_assets, cohesion) in enumerate(top_3, start=1):
        print(f"{idx}. Esperimento: {exp_id} - Numero asset: {num_assets}, Coesione: {cohesion:.4f}")

    # --- Plot dei risultati ---
    x = np.arange(len(experiment_ids))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar plot for Number of Assets
    color1 = 'tab:blue'
    ax1.set_xlabel('Experiment (custom_id)')
    ax1.set_ylabel('Number of Assets', color=color1)
    ax1.bar(x - 0.2, asset_counts, width=0.4, color=color1, label='Number of Assets')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiment_ids, rotation=45, ha="right")

    # Secondary axis for Cohesion (Average Distance to Centroid)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cohesion (Average Distance to Centroid)', color=color2)
    ax2.plot(x + 0.2, cohesion_values, color=color2, marker='o', linestyle='-', linewidth=2, label='Cohesion')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Adjust layout for readability
    fig.tight_layout()
    plt.title("Extracted Assets and Macro-cluster Cohesion per Experiment")

    # Combined Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    # Display plot
    plt.show()

    # # Grafico a barre per il numero di asset
    # color1 = 'tab:blue'
    # ax1.set_xlabel('Esperimento (custom_id)')
    # ax1.set_ylabel('Numero di asset', color=color1)
    # ax1.bar(x - 0.2, asset_counts, width=0.4, color=color1, label='Numero di asset')
    # ax1.tick_params(axis='y', labelcolor=color1)
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(experiment_ids, rotation=45, ha="right")

    # # Asse secondario per la coesione
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel('Coesione (distanza media al centroide)', color=color2)
    # ax2.plot(x + 0.2, cohesion_values, color=color2, marker='o', linestyle='-', linewidth=2, label='Coesione')
    # ax2.tick_params(axis='y', labelcolor=color2)

    # fig.tight_layout()
    # plt.title("Asset estratti e coesione del macro-cluster per esperimento")
    # # Legenda combinata
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    # plt.show()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilizzo: python script.py <ID>")
        sys.exit(1)
    file_id = sys.argv[1]
    main(file_id)
