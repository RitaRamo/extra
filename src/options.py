import argparse

from toolkit.models.lxmert_gpt_cls_vl import LxMertGPTCLSVLModel

abbr2class = {
              "lxmert_gpt_cls_vl": LxMertGPTCLSVLModel,
            }

def check_args(args):
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    add_model_args(parser)
    parsed_args = parser.parse_args(args)
    return parsed_args


def add_training_args(parser):
    group = parser.add_argument_group("Training")
    group.add_argument('--seed', default=1, type=int,
                        help="Seed")
    group.add_argument('--print-freq', default=100, type=int,
                        help="TODO")
    group.add_argument('--workers', default=1, type=int,
                        help="TODO")
    group.add_argument("--cpu", action="store_true")

    group.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    group.add_argument("--max-epochs", type=int, default=120,
                        help="Maximum number of training epochs")
    group.add_argument("--epochs-early-stopping", type=int, default=5)

    group.add_argument("--max-caption-len", type=int, default=50)


    #optimization args
    group.add_argument("--criterion", default="cross_entropy")
    group.add_argument("--regularization", default=None)
    group.add_argument("--generation-criterion", default="cross_entropy")
    group.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "adamW", "radam"])
    group.add_argument("--lr_scheduler", action="store_true")
    group.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clip")

    #dataset args
    group.add_argument("--image-features-filename",
                        help="Folder where the preprocessed data is located")
    group.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the dataset splits")

    #checkpoint args
    group.add_argument("--checkpoints-dir", default=None,
                       help="Path to checkpoint of previously trained model")
    group.add_argument("--checkpoints-critical-dir", default=None,
                       help="Path to checkpoint of critical trained model")

    group.add_argument("--debug", action="store_true")
    group.add_argument("--critical", action="store_true")
    group.add_argument("--critical_greedy", action="store_true")
    group.add_argument("--reward_with_bleu", action="store_true")
    group.add_argument("--restart_optim", action="store_true")

    return group


def add_model_args(parser):
    group = parser.add_argument_group("Model")
    group.add_argument("--fine-tune-encoder", action="store_true",
                        help="Fine tune the encoder")
    # Add model-specific arguments to the parser
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True

    for k, v in abbr2class.items():
        model_parser = subparsers.add_parser(k)
        v.add_args(model_parser)
    
    return group
