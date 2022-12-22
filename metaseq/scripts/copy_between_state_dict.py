import torch
import fire
import os

def copy_between_state_dict(source_model_path, target_model_path, output_model_path, key="shard_metadata"):
    print(f"Copy {key} from {source_model_path} to {target_model_path} and save to {output_model_path}")
    sd = torch.load(source_model_path, map_location="cpu")
    tsd = torch.load(target_model_path, map_location="cpu")
    tsd[key] = sd.pop(key)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(tsd, output_model_path)
    print(f"Saved {output_model_path}!")
    return True

if __name__ == "__main__":
    fire.Fire(copy_between_state_dict)