import os
import json
import requests
import argparse
from tqdm import tqdm
from statistics import mean

from torch.utils.data import DataLoader, SequentialSampler
from metaseq.data import JsonlDataset
from metaseq.generation_metrics.parlai_utils import DialogMetrics

# url = "http://localhost:5000/evaluate"
# headers = {"Content-Type": "application/json"}
# data = {"text": "The quick brown fox jumps over the lazy dog."}
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.json())

def json_collate_fn(batch):
    return {
        "prompt": [x['prompt'] for x in batch],
        "label": [x['label'] for x in batch],
    }

def line_processor(json_line):
    prompt, label = json_line["text"].split("<s>")
    prompt += "<s>"
    return {
        "prompt": prompt,
        "label": label,
    }

def generate_predictions(args):
    assert args.input_file.endswith(".jsonl")
    dataset = JsonlDataset(
        path=args.input_file,
        tokenizer=line_processor,
        epoch=1,
        data_subshard_count=1,
    )
    dataloader = DataLoader(dataset,
                    batch_size=args.batch_size, 
                    collate_fn=json_collate_fn, 
                    sampler=SequentialSampler(dataset),
                    num_workers=8,
                )

    request_data_template = {
        "prompt": [],
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logprobs": args.logprobs,
        "n": args.n,
        "best_of": args.best_of,
        "echo": args.echo,
    }
    url = f"http://{args.host}:{args.port}/completions"
    with open(args.prediction_file, "w", encoding="utf-8") as out_f:
        for _, batch in enumerate(tqdm(dataloader)):
            request_data_template['prompt'] = batch['prompt']
            response = requests.post(url, json=request_data_template)
            response = response.json()

            for i in range(0, len(response['choices']), args.n):
                predictions = response['choices'][i:i+args.n]
                row = {
                    "predictions": [p['text'] for p in predictions],
                    "label": batch['label'][i//args.n],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.close()

def calculate_generation_metrics(args):
    metrics = DialogMetrics(metrics_list=args.metrics_list)
    average_metrics = {}
    with open(args.prediction_file, "r", encoding="utf-8") as in_f, open(args.metrics_file, "w", encoding="utf-8") as out_f:
        for line in in_f:
            row = json.loads(line)
            m = metrics.evaluate_response(row['predictions'][0], [row['label']])
            out_f.write(json.dumps(m, ensure_ascii=False) + "\n")
            
            for k,v in m.items():
                if k not in average_metrics:
                    average_metrics[k] = []
                average_metrics[k].append(v)

        for k,v in average_metrics.items():
            average_metrics[k] = mean(v)
        out_f.write(json.dumps(average_metrics, ensure_ascii=False) + "\n")
        in_f.close()
    out_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=46010)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--prediction-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)

    # Generation Parameters
    parser.add_argument("--min-tokens", type=int, default=1, help="blocks EOS until at least this many tokens is provided")
    parser.add_argument("--max-tokens", type=int, default=512, help="forces EOS after this many tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="softmax temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus probability")
    parser.add_argument("--logprobs", type=int, default=0, help="return this cutoff of the probability distribution")
    parser.add_argument("--n", type=int, default=1, help="number of beams to return. must be <= best_of")
    parser.add_argument("--best-of", type=int, default=1, help="beam size")
    parser.add_argument("--echo", type=bool, default=False, help="if true, returned text/tokens/scores includes the prompt.")

    # Metrics
    parser.add_argument("--metrics-list", type=str, default="bleu,rouge", help="comma separated list of metrics to calculate")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    
    generate_predictions(args)
    calculate_generation_metrics(args)