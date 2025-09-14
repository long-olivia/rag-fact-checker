import os
import re
import numpy as np
import json

def doc_grouper(folder_dir):
    base_dir=folder_dir
    docs=[]
    tag="doc"
    for file in os.listdir(folder_dir):
        file_path=os.path.join(base_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content=f.read()
            reg_str = r"<doc[^>]*>(.*?)</doc>"
            res = re.findall(reg_str, content, flags=re.S)
            docs.append(res)
    with open("results.json", 'w') as file:
        json.dump(docs, file, ensure_ascii=False)
    return docs
 

# def sectionizer(folder_dir):
#     docs = doc_grouper(folder_dir)
#     for txt_file in docs:
#         for article in txt_file:


#     sections=[]
#     for file in os.listdir(folder_dir):
#         file_path = os.path.join(base_dir, file)
#         with open(file_path, 'r', encoding="utf-8") as f:
#             lines=[line.strip() for line in f if line.strip()]
#             print(lines)
        

if __name__ == "__main__":
    doc_grouper("./text")