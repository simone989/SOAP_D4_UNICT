import os
import json
import re
import argparse
from openai import OpenAI
from dotenv import load_dotenv

def extract_json_from_content(content: str):
    """
    Given a content string that may be wrapped in triple backticks with a "json" label,
    extract the inner JSON text and parse it.
    """
    content = content.strip()
    # Pattern to capture JSON between ```json and ```
    pattern = r"^```json\s*(.*?)\s*```$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = content
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Error parsing JSON content:", e)
        return None

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

def extract_content_json(input_filename: str) -> list:
    """
    Reads the JSONL file line by line and, for each response,
    extracts and parses the JSON embedded in the first choice's message.content.

    Returns a list of the parsed JSON objects.
    """
    extracted = []
    with open(input_filename, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
                # Navigate to the message content of the first choice
                choices = obj.get("response", {}).get("body", {}).get("choices", [])
                if choices and isinstance(choices, list):
                    content = choices[0].get("message", {}).get("content", "")
                    parsed = extract_json_from_content(content)
                    if parsed is not None:
                        extracted.append(parsed)
                else:
                    print("No choices found in response.")
    return extracted

def save_results(results: dict, output_filename: str):
    """
    Saves the results JSON object to disk.
    """
    try:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
        print(f"Extracted results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download batch results using batch_id, extract the JSON embedded in each response's content, and save the consolidated results to a file."
    )
    parser.add_argument("batch_id", help="The batch ID from which to download the results.")
    parser.add_argument("output_file", help="Filename for the consolidated extracted JSON output.")
    args = parser.parse_args()

    load_dotenv()  # Load environment variables (e.g., OPENAI_API_KEY)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Retrieve the raw batch results JSONL file.
    raw_results_file = retrieve_batch_results(args.batch_id, client)

    # Extract the JSON from the "content" field of each response.
    extracted_json = extract_content_json(raw_results_file)

    # Consolidate results in a dict.
    results = {"results": extracted_json}

    # Save the extracted results to the output file.
    save_results(results, args.output_file)

    # Also print the consolidated results.
    print(json.dumps(results, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
