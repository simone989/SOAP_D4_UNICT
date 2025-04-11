import fitz  # PyMuPDF
import re

START_PAGE = 3

def merge_text_blocks(page_text_block):
    merged_text = []
    current_block = ""
    for block in page_text_block:
        if re.match(r"^\(\w+\)", block):
            current_block += re.sub(r"^\(\w+\)", "", block)
        elif re.match(r"^\d+\.", block) or re.match(r"^\d+\.\d+\.", block):
            current_block += re.sub(r"^\d+\.", "", block)
        elif block.endswith(";") or block.endswith(".") or block.endswith(":"):
            current_block += block
            merged_text.append(current_block)
            current_block = ""
        else:
            current_block += block + " "
    if current_block:
        merged_text.append(current_block)
    return "".join(merged_text)

def extract_text(pdf_path, footer_height=200, start_page = START_PAGE):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(start_page,  len(doc)): ##len(doc)
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height

        filtered_blocks = [
            block for block in blocks
            if block["bbox"][1] < (page_height - footer_height)
        ]

        page_text_block = [
            "".join([span["text"] for line in block["lines"] for span in line["spans"]])
            for block in filtered_blocks
        ]
        merged_text_blocks = merge_text_blocks(page_text_block)
        full_text += merged_text_blocks

    return full_text

def save_text_to_file(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


pdf_path = "./asset/Ai_Act/TA-9-2024-0138-FNL-COR01_EN.pdf"
output_txt = "extracted_text.txt"

# Supponendo che extract_text sia definita e restituisca una stringa unificata
full_text = extract_text(pdf_path)
save_text_to_file(full_text, output_txt)
print(f"Testo estratto salvato in {output_txt}")