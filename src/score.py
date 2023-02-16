import os
import sys
import json
import argparse
from toolkit.evaluation.metrics import coco_metrics

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    split = args.results_fn.split("/")[-1].split(".")[0]
    
    output_fn = os.path.join(args.output_dir,
                                "coco" + "." + ".".join(args.results_fn.split("/")[-1].split(".")[1:-1]) + "." + split)
    
    metric2score, each_image_score = coco_metrics(args.results_fn, args.annotations_dir, args.annotations_split)
    with open(output_fn, "w") as f:
        for m, score in metric2score.items():
            f.write("%s: %f\n" % (m, score))

    with open(output_fn+"individual_scores.json", 'w+') as f:
        json.dump(each_image_score, f, indent=2)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-fn", help="Path to JSON file of image -> ground-truth captions.")
    parser.add_argument("--results-fn", help="Path to JSON file of image -> caption output by the model.")
    parser.add_argument("--top-results-fn", help="Path to JSON file of image -> top-k captions output by the model. Used for recall.")
    parser.add_argument("--output-dir",
                        help="Directory where to store the results.")

    # COCO
    parser.add_argument("--annotations-dir",
                        help="Path to COCO 2014 trainval annotations directory.")
    parser.add_argument("--annotations-split", choices=["train2014", "val2014", "test2014"],
                        help="COCO 2014 trainval annotations split.")

    parsed_args = parser.parse_args(args)
    return parsed_args 


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
