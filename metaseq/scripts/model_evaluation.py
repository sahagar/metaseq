# Send json request to the model evaluation server

import requests
import argparse
import os
import itertools

from metaseq.data import JsonlDataset

# url = "http://localhost:5000/evaluate"
# headers = {"Content-Type": "application/json"}
# data = {"text": "The quick brown fox jumps over the lazy dog."}
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.json())

def read_chunks(dataset, bz=16):
    for i in range(0, len(dataset), bz):
        yield dataset[i:i+bz]

def generate_predictions(args):
    assert args.input_file.endswith(".jsonl")
    dataset = JsonlDataset(
        path=args.input_file,
        tokenizer=None,
        epoch=1,
        data_subshard_count=1,
    )

    with open(args.prediction_file, "w", encoding="utf-8") as out_f:
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i:i+args.batch_size]
            print(batch[0])
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=46010)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--prediction-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    
    generate_predictions(args)