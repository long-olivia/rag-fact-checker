import json

def chunk():
    chunk_id=0
    result=[]
    with open("categorized.json", 'r') as f:
        data=json.load(f)
    for article in data:
        keys=iter(article)
        key=next(keys, "end")
        title=key
        while True:
            if key=="end":
                break
            s=article[key]
            splits = [s[i:i+500] for i in range(0, len(s), 500)]
            for x in splits:
                chunk_dict={}
                chunk_dict["ID"]=chunk_id
                chunk_dict["Title"]=title
                chunk_dict["Section"]=key
                chunk_dict["Content"]=x
                chunk_id+=1
                result.append(chunk_dict)
            key=next(keys, "end")

    with open("chunks.json", 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

            

if __name__ == "__main__":
    chunk()