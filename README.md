# Scripts Overview

This repository contains two main groups of scripts under the `scripts/` folder: **Assets Extraction** and **Threat Elicitation**. Each group implements a series of processes that support the extraction, processing, and analysis of legal texts and threat information using AI (OpenAI API) and data processing techniques.

**Methodology:**
The implementation of these scripts is based on a structured methodological framework that combines natural language processing (NLP) techniques, machine learning for clustering and vector computations, and batch processing strategies. This methodology is implemented by these scripts to ensure comprehensive analysis and consolidation of asset and threat data extracted from legal texts.

---

## Assets Extraction

- <a id="extract-asset"></a> **extract_asset.py**
  This script loads legal text documents and extracts nouns using spaCy. It then computes noun vectors, runs clustering experiments (varying thresholds for cluster selection and merging), and prepares a batch of requests. These requests are sent to the AI API to extract structured asset clusters from the legal text.

- <a id="read-extract-asset-result"></a> **read_extract_asset_result_by_batch.py**
  This script processes the raw batch results from the AI API. For each result, it parses the response JSON, applies duplicate removal (using NLP lemmatization) within each cluster, and then updates an intermediate JSON file with the final clusters. The updated results are saved for later review or further analysis.

- <a id="plot-result"></a> **plot_result.py**
  After the experiments have been executed, this script loads the results and plots key metrics. It displays the number of extracted assets and the cohesion of clusters (average distance to the centroid) per experiment. The visual output helps to identify the top experiments based on asset extraction quality.

---

## Threat Elicitation

- <a id="extract-threat-with-batch"></a> **extract_threat_with_batch.py**
  This script is responsible for processing legal text files to identify potential security threats. It prepares messages for the AI API using given instructions, creates batched requests (multiple executions per file), and sends these requests to the API. The responses are then used to extract threat-related information from the legal articles.

- <a id="read-threat-from-batch"></a> **read_threat_from_batch.py**
  This script retrieves batch results specific to threat elicitation. It extracts the necessary threat information from the responses, aggregates the data, and saves the final threat summary into an output file.

- <a id="group-threat-by-articles-runs"></a> **group_threat_by_articles_runs.py**
  This script aggregates threat information by grouping the responses according to legal articles. It reads aggregated batch responses and produces an output JSON that summarizes the threats associated with each article.

- <a id="merge-similar-threat-by-articles"></a> **merge_similar_threat_by_articles.py**
  This script further processes the threat data by merging similar threat entries based on their associated articles. It produces a consolidated output where redundant or similar threat items are merged to simplify analysis.

- <a id="merge-similar-threat"></a> **merge_similar_threat.py**
  This script performs a similar consolidation process for threat data. It transforms the merged results and creates one or more batch entries to update the threat analysis by merging similar threat entries from the raw batch results.

- <a id="get-final-threat"></a> **get_final_threat.py**
  This script downloads the batch results using a provided batch ID. It extracts the embedded JSON from the content field of each response, consolidates the threat information, and saves the final, aggregated threat data to a specified file.

---

## Workflow for Applying the Methodology

1. **Preparation & Data Collection**
   - Gather the legal text documents to be analyzed.
   - Ensure the texts are accessible as expected.

2. **Assets Extraction Workflow**
   - **Extraction:**
     - Run `extract_asset.py` to load legal texts, extract nouns using spaCy, compute noun vectors, and carry out clustering experiments.
     - The script prepares batched requests that are submitted to the AI API to extract asset clusters.
   - **Post-Processing:**
     - Use `read_extract_asset_result_by_batch.py` to process the raw API responses. The script parses the JSON content, removes duplicates with NLP lemmatization, and updates an intermediate JSON file with the final clusters.
   - **Visualization:**
     - Execute `plot_result.py` to visualize key metrics like the number of extracted assets and cluster cohesion, aiding in the identification of top-performing experiments.

3. **Threat Elicitation Workflow**
   - **Threat Extraction:**
     - Run `extract_threat_with_batch.py` to process legal texts for potential security threats. This script prepares and sends batched API requests for threat identification.
   - **Aggregation & Grouping:**
     - Use `read_threat_from_batch.py` to extract and aggregate threat details from API responses.
     - Run `group_threat_by_articles_runs.py` to group threats by legal articles, creating an initial organized summary.
   - **Consolidation:**
     - Execute `merge_similar_threat_by_articles.py` to merge similar threat entries grouped by articles.
     - Run `merge_similar_threat.py` for further consolidation of batch results.
   - **Finalization:**
     - Run `get_final_threat.py` to download and compile the comprehensive threat data using a specified batch ID, yielding the final aggregated threat summary.

---

This README provides a high-level overview of each script’s function and the workflow to apply the underlying methodology. Use the appropriate scripts based on whether you wish to extract assets from legal texts or analyze potential threats described within those texts.