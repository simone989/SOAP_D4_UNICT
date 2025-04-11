import os
import json
import re
import argparse
from openai import OpenAI
from dotenv import load_dotenv

def get_client():
    """
    Initializes the OpenAI client using the API key from environment variables.
    """
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client

def retrieve_batch_results(batch_id: str, client) -> str:
    """
    Retrieves the content of the results file associated with the given batch_id.
    Assumes that the detailed batch info contains an 'output_file_id' field.
    Writes the raw content to a file and returns the text content.
    """
    # Retrieve detailed batch information
    batch_details = client.batches.retrieve(batch_id)
    result_file_id = batch_details.output_file_id
    # Retrieve the file content using the result file ID
    content = client.files.content(result_file_id)
    text_content = content.text

    output_filename = f"raw_batch_results_{batch_id}.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"Raw batch results written to {output_filename}")
    return output_filename

def extract_json_from_content(content: str):
    """
    Extracts the inner JSON from a content string.
    It removes Markdown code block markers (e.g., "```json" and "```") and parses the JSON.
    """
    # Use regex to capture everything between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If not formatted as a code block, use the whole content string stripped of whitespace
        json_str = content.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Error parsing JSON content:", e)
        return None

def process_response_file(input_filename: str):
    """
    Reads the input JSONL file line by line.
    For each response, extracts the custom_id and the JSON result from the first choice in the response.
    Aggregates all extracted results into a list.
    """
    aggregated_results = []
    with open(input_filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    response_obj = json.loads(line)
                    custom_id = response_obj.get("custom_id")
                    choices = response_obj.get("response", {}).get("body", {}).get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        parsed_json = extract_json_from_content(content)
                        result_entry = {
                            "custom_id": custom_id,
                            "analysis": parsed_json
                        }
                        aggregated_results.append(result_entry)
                except json.JSONDecodeError as e:
                    print("Error decoding line:", e)
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve batch results by batch_id, extract JSON analysis, and aggregate results."
    )
    parser.add_argument("batch_id", help="The batch ID to retrieve the results for.")
    parser.add_argument("output_file", help="Output JSON file to write aggregated results.")
    args = parser.parse_args()

    client = get_client()

    # Retrieve the raw batch results file using the provided batch_id.
    raw_results_file = retrieve_batch_results(args.batch_id, client)

    # Process the raw JSONL file and aggregate the results.
    aggregated_results = process_response_file(raw_results_file)

    # Write the aggregated results to the output file.
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(aggregated_results, outfile, indent=4, ensure_ascii=False)
    print(f"Aggregated results written to {args.output_file}")

if __name__ == "__main__":
    main()
