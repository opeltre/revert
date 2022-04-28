import argparse
import os

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, nargs=1, help="input state name => '$REVERT_MODELS/{in}.pt' (model randomly initialized otherwise)", metavar="file_name_without_extension")
    parser.add_argument('--output', '-o', type=str, nargs=1, help="output state name => '$REVERT_MODELS/{out}.pt' and creates tensorboard traces in '$REVERT_LOGS/{out}' (out = 'convnet-apr28-1' by default for instance)", metavar="file_name_without_extension")
    return parser.parse_args()

def arg_verifier(args):
    if not 'REVERT_MODELS' in os.environ:
        raise OSError("The REVERT_MODELS environment variable is not defined.")
    if not 'REVERT_LOGS' in os.environ:
        raise OSError("The REVERT_LOGS environment variable is not defined.")

    if args.input:
        path_input = os.environ['REVERT_MODELS'] + "/" + args.input[0] + ".pt"
        print("Using", path_input, "as input")
    else:
        print("Model randomly initialized")

    if args.output:
        path_output = os.environ['REVERT_MODELS'] + "/" + args.output[0] + ".pt"
        print("Using", path_output, "as output")

        path_traces = os.environ['REVERT_LOGS'] + "/" + args.output[0] + "/"
        print("Using", path_traces, "as traces")
    else:
        print("Using default output (convnet-...)")

args = arg_parser()
arg_verifier(args)
