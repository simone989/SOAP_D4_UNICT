import json
import argparse

def merge_threats(threats1, threats2):
    """
    Merge two lists of threat dictionaries (each with 'threat' and 'explanation').
    Duplicate threats (by threat name) are avoided.
    """
    merged = { threat["threat"]: threat for threat in threats1 }
    for threat in threats2:
        key = threat["threat"]
        if key not in merged:
            merged[key] = threat
    return list(merged.values())

def merge_threat_regulation_lists(global_list, new_list):
    """
    Merge a new threat_regulation_list (a list of dicts, each with "asset" and "threats")
    into a global dictionary mapping asset to list of threat dictionaries.
    """
    for entry in new_list:
        asset = entry.get("asset")
        if not asset:
            continue
        new_threats = entry.get("threats", [])
        if asset not in global_list:
            global_list[asset] = new_threats
        else:
            global_list[asset] = merge_threats(global_list[asset], new_threats)
    return global_list

def aggregate_articles(aggregated_results):
    """
    Aggregates threat regulation lists across all responses grouped by article_id.
    The final output is a JSON object with an "articles" key, whose value is a list
    of objects. Each object has an "article_id" and its merged "threat_regulation_list".
    """
    articles_agg = {}
    for entry in aggregated_results:
        analysis = entry.get("analysis")
        if not analysis:
            continue
        for article in analysis.get("articles", []):
            article_id = article.get("article_id")
            if not article_id:
                continue
            new_threat_list = article.get("threat_regulation_list", [])
            if article_id not in articles_agg:
                articles_agg[article_id] = merge_threat_regulation_lists({}, new_threat_list)
            else:
                articles_agg[article_id] = merge_threat_regulation_lists(articles_agg[article_id], new_threat_list)

    final_articles = []
    for article_id, asset_dict in articles_agg.items():
        threat_entries = []
        for asset, threats in asset_dict.items():
            threat_entries.append({
                "asset": asset,
                "threats": threats
            })
        final_articles.append({
            "article_id": article_id,
            "threat_regulation_list": threat_entries
        })
    return {"articles": final_articles}

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate threat information by article from aggregated batch responses."
    )
    parser.add_argument("input_file", help="Input JSON file containing aggregated responses (list of dicts).")
    parser.add_argument("output_file", help="Output JSON file for aggregated article threats.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as infile:
        aggregated_results = json.load(infile)

    final_output = aggregate_articles(aggregated_results)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(final_output, outfile, indent=4, ensure_ascii=False)
    print(f"Aggregated article threat information written to {args.output_file}")

if __name__ == "__main__":
    main()
