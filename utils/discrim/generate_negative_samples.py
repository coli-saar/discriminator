from tqdm import tqdm
import numpy as np
from utils.io.io import read_file, write_file, mkdir, read_json, read_jsonl, glob
np.random.seed(42)

def generate_model_negatives(dirpath, tgt_dirpath, negative_ratio=1.0,):
    filename2lines = {"train.tsv": [f"{dir}/out.train.pred" for dir in glob(dirpath, 'output_e*')],}

    for f in filename2lines:
        lines = []
        for filepath in filename2lines[f]:
            json_objs = read_jsonl(filepath)
            for json_obj in tqdm(json_objs):
                beam_size = len(json_obj["beam_predicted_text"]) if "beam_predicted_text" in json_obj else 1
                batch_size = len(json_obj["metadata"])
                for i in range(batch_size):
                    predicted_texts = [json_obj["beam_predicted_text"][j][i] for j in range(beam_size)] if "beam_predicted_text" in json_obj \
                        else [json_obj["predicted_text"][i]]
                    target_text = json_obj["metadata"][i]["target_text"]
                    source_text = json_obj["metadata"][i]["source_text"]
                    np.random.shuffle(predicted_texts)
                    neg_num = 0
                    for predicted_text in predicted_texts:
                        if predicted_text != target_text:
                            negative_example = f"{source_text} | {predicted_text}\tFalse\n"
                            lines.append(negative_example)
                            neg_num += 1
                            if neg_num >= negative_ratio:
                                break
                    lines.append(f"{source_text} | {target_text}\tTrue\n")
        tgt_filepath = f"{tgt_dirpath}/{f}"
        write_file(tgt_filepath, lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("tgt_dirpath", type=str, default=None)
    parser.add_argument("--negative_ratio", type=float, default=1.0)

    args = parser.parse_args()
    mkdir(args.tgt_dirpath)
    func_args = {"dirpath": args.dirpath,
                 "tgt_dirpath": args.tgt_dirpath,
                    "negative_ratio": args.negative_ratio,
                 }
    generate_model_negatives(**func_args)

if __name__ == "__main__":
    main()