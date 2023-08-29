import json
import re

# 从txt文件读取数据
with open('rwkv_vocab_v20230424.txt', 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    id_ = int(parts[0])
    
    # 使用正则表达式找到最后一个数字的位置
    last_num_match = re.search(r"(\d+)$", line)
    if last_num_match:
        last_num_start = last_num_match.start()
        # 使用切片操作提取原始的字符串
        content = line[len(str(id_)) + 1:last_num_start].strip().strip("'")
        data.append((id_, content, 1))

# 转换为HuggingFace tokenizer格式
tokenizer_format = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": {"type": "NFC"},
    "pre_tokenizer": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True
    },
    "post_processor": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True
    },
    "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True
    },
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": None,
        "merges": []
    }
}

vocab = []

for item in data:
    token_entry = {
        "id": item[0],
        "content": item[1],
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": False
    }
    if item[1] == "0":
        token_entry['special'] = True
    tokenizer_format["added_tokens"].append(token_entry)
    vocab.append(item[1])

result_vocab = {word: index for index, word in enumerate(vocab)}
tokenizer_format['model']['vocab'] = result_vocab

# 将结果写入json文件
with open('tokenizer.json', 'w', encoding='utf-8') as json_file:
    json.dump(tokenizer_format, json_file, ensure_ascii=False, indent=4)
