import os
import json
import re
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Placeholder instructions (to be refined later)
instructions = """
You are an expert in threat elicitation, spanning multiple domains including cybersecurity, legal regulations, operational risks, and more. Your task is to analyze the threat information extracted from multiple articles and produce a single, consolidated list of threat entries.

Input:
- You will receive a JSON input that contains threat data from multiple articles. Each entry includes one or more assets with their associated threats.
- Additionally, you will receive the full legal text from which these threats were extracted. This text provides extra context to help you understand the source document.


Your goal is to ignore the original asset grouping and generate one comprehensive list of threat entries by following these steps:
1.Review all threat entries across the input data.
2.Identify duplicate or very similar threats, regardless of the asset they were originally assigned to.
3.Merge similar threats into a single, concise entry that captures the essential risk and explanation.
4.Remove any duplicate threat entries.
5.Produce a final JSON output that consists of a single array of threat entries. Each entry should contain only:
   - "threat": the consolidated threat name.
   - "explanation": the merged explanation for the threat.

Example output format (compact):
{"threats":[{"threat":"Data Breach","explanation":"High-risk systems handling sensitive data may lead to unauthorized access and breaches."},{"threat":"Data Manipulation","explanation":"Improper data handling may result in manipulation, leading to incorrect outputs."},{"threat":"Unauthorized Access","explanation":"Robust security measures are necessary to prevent unauthorized access to critical systems."}]}

Your goal is to provide a concise and comprehensive consolidated list of threats, ensuring that all key risks across all contexts are represented without redundancy. Remember, you are performing threat elicitation: the process of identifying potential threats and vulnerabilities to assess risks and plan effective countermeasures.
"""


def read_legal_text(file_path: str) -> str:
    """
    Reads and returns the full legal text from the provided file path.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            return infile.read().strip()
    except Exception as e:
        print(f"Error reading legal text from {file_path}: {e}")
        return ""

def extract_json_from_content(content: str, custom_id: str) -> dict:
    """
    Given a content string that may be wrapped in triple backticks with a "json" label,
    extract the inner JSON text and parse it.
    """
    content = content.strip()
    pattern = r"^```json\s*(.*?)\s*```$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = content
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content for custom_id {custom_id}: {e}")
        return {}

def retrieve_batch_results(batch_id: str, client) -> str:
    """
    Retrieves the content of the results file associated with the given batch_id.
    Assumes that the batch details contain an 'output_file_id' field.
    Saves the raw JSONL content to a file and returns the filename.
    """
    batch_details = client.batches.retrieve(batch_id)
    result_file_id = batch_details.output_file_id
    content = client.files.content(result_file_id)
    text_content = content.text

    output_filename = f"raw_batch_results_{batch_id}.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"Raw batch results saved to {output_filename}")
    return output_filename

def process_jsonl_file(input_filename: str) -> dict:
    """
    Reads a JSONL file where each line is a JSON object representing a batch response.
    For each response, extracts the id, custom_id, and parses the message content (from the first choice)
    as JSON (removing Markdown code block markers if present).
    Returns a merged JSON object with a key "results" containing a list of:
    {
        "id": <response id>,
        "custom_id": <custom id>,
        "parsed_content": <parsed JSON object from message content>
    }
    """
    merged = []
    with open(input_filename, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
                entry_id = obj.get("id", "")
                custom_id = obj.get("custom_id", "")
                choices = obj.get("response", {}).get("body", {}).get("choices", [])
                if choices and isinstance(choices, list):
                    message_content = choices[0].get("message", {}).get("content", "")
                    parsed_content = extract_json_from_content(message_content, custom_id)
                else:
                    parsed_content = None
                new_obj = {
                    "id": entry_id,
                    "custom_id": custom_id,
                    "parsed_content": parsed_content
                }
                merged.append(new_obj)
    return {"results": merged}

def parse_article_from_custom_id(custom_id: str) -> str:
    """
    Extracts the article reference from a custom_id.
    Assumes the custom_id is formatted like: "batch-Article 6-exec-1-1740747182".
    Returns "Article 6" in that case.
    """
    parts = custom_id.split("-")
    if len(parts) >= 2:
        return parts[1]
    return ""

def transform_results(merged_data: dict) -> dict:
    """
    Transforms the merged JSON data (with key "results") into a single consolidated threat list.
    Each entry in the input has a "parsed_content" field containing an "assets" list.
    This function iterates over each entry, extracts the "assets" list, and then merges all threat entries,
    grouping them by asset and avoiding duplicates.

    The final output format will be:
    {
        "consolidated_threats": [
            {"asset": "USER", "threats": [{"threat": "Threat Name", "explanation": "Merged explanation"}, ...]},
            {"asset": "SOFTWARE", "threats": [ ... ]},
            ...
        ]
    }
    """
    merged_threats = {}
    for entry in merged_data.get("results", []):
        parsed_content = entry.get("parsed_content", {})
        assets = parsed_content.get("assets", [])
        for item in assets:
            asset = item.get("asset")
            threats = item.get("threats", [])
            if not asset:
                continue
            if asset not in merged_threats:
                merged_threats[asset] = threats.copy()
            else:
                for threat_entry in threats:
                    threat_name = threat_entry.get("threat")
                    if threat_name and not any(t.get("threat") == threat_name for t in merged_threats[asset]):
                        merged_threats[asset].append(threat_entry)
    consolidated = []
    for asset, threats in merged_threats.items():
        consolidated.append({
            "asset": asset,
            "threats": threats
        })
    return {"consolidated_threats": consolidated}


def save_merged_json(merged_data: dict, output_filename: str):
    """
    Saves the merged JSON object to disk.
    """
    try:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
        print(f"Merged JSON saved to {output_filename}")
    except Exception as e:
        print(f"Error saving merged JSON: {e}")

def prepare_messages(transformed_data: dict, legal_text: str) -> list:
    """
    Prepares the messages payload for the batch API request.
    The assistant message contains the instructions.
    (Optionally, the legal text can be included in another assistant message.)
    The user message contains the JSON-formatted transformed data in compact form,
    but with the last 5% of the data removed.
    """
    compact_data = json.dumps(transformed_data, separators=(',',':'))
    # cutoff = int(len(compact_data) * 0.92)
    # compact_data = compact_data[:cutoff]
    return [
        {"role": "assistant", "content": instructions},
        {"role": "assistant", "content": legal_text},
        {"role": "user", "content": compact_data}
    ]


def create_single_batch_entry(transformed_data: dict, legal_text: str, num_executions: int = 1) -> list:
    """
    Creates one or more batch entries that contain the entire transformed JSON data along with the legal text.
    This function creates 'num_executions' copies of a single entry, each with the same content but a unique custom_id.
    """
    entries = []
    for exec_num in range(1, num_executions + 1):
        custom_id = f"batch-all-exec-{exec_num}-{int(time.time())}"
        entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "o1-mini",  # Adjust model name if necessary.
                "messages": prepare_messages(transformed_data, legal_text)
            },
        }
        entries.append(entry)
    return entries

def save_entries_jsonl(entries: list, output_filename: str):
    """
    Saves a list of batch entries to a JSONL file (one JSON object per line).
    """
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for entry in entries:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + "\n")
        print(f"Batch entries saved to {output_filename}")
    except Exception as e:
        print(f"Error saving JSONL file: {e}")

def invoke_next_batch(batch_entry_filename: str, client):
    """
    Placeholder function that uploads the batch entry file and creates a new batch job.
    Detailed instructions for the batch call will be defined later.
    """
    try:
        with open(batch_entry_filename, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        batch_input_file_id = batch_input_file.id
        print(f"Batch entry file uploaded with ID: {batch_input_file_id}")
        batch_response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",  # Update if necessary.
            completion_window="24h",
            metadata={
                "description": f"Batch processing for merged results from {batch_entry_filename}. Instructions TBD."
            }
        )
        batch_id = batch_response.id
        print(f"Next batch created with ID: {batch_id}")
        return batch_id
    except Exception as e:
        print(f"Error invoking next batch: {e}")
        return None

def main():
    batch_entry_file = "batch_request.jsonl"
    legal_text_file = "./asset/Nis_2/nis2.txt"  # Modifica il percorso della cartella

    parser = argparse.ArgumentParser(
        description="Process an input JSON file containing merged results, transform the data to consolidate threat entries, add legal text context, create batch entries from the transformed data (possibly multiple copies), save the entries in JSONL format, and optionally invoke the next batch job."
    )
    parser.add_argument("batch_id", help="Input JSON file with merged results (from previous steps).")
    parser.add_argument("merged_output_file", help="Filename for the merged and transformed JSON output.")
    parser.add_argument("--invoke", action="store_true", help="If set, invoke the next batch job using the batch entry file.")
    args = parser.parse_args()

    load_dotenv()  # Load API key and other variables from .env
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Step 1: Retrieve the raw batch results file using the provided batch_id.
    raw_results_file = retrieve_batch_results(args.batch_id, client)
    # Read the full legal text from the provided file path.
    legal_text = read_legal_text(legal_text_file)

    # Step 2: Process the raw JSONL file and merge its contents, parsing the message content as JSON.
    merged_data = process_jsonl_file(raw_results_file)
    #print(merged_data)
    # Step 13: Transform the input data using the new structure.
    transformed_data = transform_results(merged_data)

    # Step 4: Save the transformed JSON to disk.
    save_merged_json(merged_data, args.merged_output_file)

    # Step 5: Create a single batch entry (or multiple identical ones) from the entire transformed JSON,
    #         including the legal text as additional context.
    batch_entries = create_single_batch_entry(transformed_data, legal_text, num_executions=1)

    # Step 6: Save the batch entry(ies) in JSONL format.
    save_entries_jsonl(batch_entries, batch_entry_file)

    # Step 7: Optionally, invoke the next batch using the batch entry file.
    if args.invoke:
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        invoke_next_batch(batch_entry_file, client)

if __name__ == "__main__":
    main()
