import os
import glob
import pickle
import tiktoken
import numpy as np

from multiprocessing.pool import ThreadPool

MIN_ARTICLE = 1024
# walk through all files in directory
input_path = "/home/robin/Downloads/text/*/*"


# remove "&lt;templateXXXXXXX&gt;"
def line_process(line):
    length = len(line)
    ans = ""
    read = True
    index = 0
    while index < length:
        if line[index: index + 4] == "&lt;":
            read = False
            index += 4
            continue
        elif line[index: index + 4] == "&gt;":
            read = True
            index += 4
            continue
        elif read:
            ans += line[index]
        index += 1
    return ans


def article_process(lines):
    start = False
    articles = []
    content = ""
    for line in lines:
        if line.startswith("<doc id="):
            start = True
        elif start:
            if line.startswith("</doc>"):
                if len(content) > MIN_ARTICLE:
                    articles.append(content)
                start = False
                content = ""
            else:
                line = line_process(line)
                content += line
    return articles


total_articles = []

all_files = list(glob.glob(input_path))
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

index = 0
files = 0

for file_name in all_files:
    with open(file_name, "r") as fp:
        print(file_name)
        articles = article_process(fp.readlines())
    files += 1
    total_articles += enc.encode_ordinary("\n\n".join(articles))

    if files == 2000:
        data = total_articles
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9):]
        train_ids, val_ids = train_data, val_data
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(os.path.dirname(__file__), f'train{index}.bin'))
        val_ids.tofile(os.path.join(os.path.dirname(__file__), f'val{index}.bin'))
        index += 1
        files = 0
        total_articles = []
