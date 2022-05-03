import argparse
import datetime
import os

def arg_parser(prefix='convnet'):
    parser = argparse.ArgumentParser()

    if prefix == 'convnet':
        parser.add_argument('--input', '-i', type=str, nargs=1, help="Should target an existing '.pt' file. If 'path' is not absolute, the model will be searched using the following path : $REVERT_MODELS/path (make sure to define the '$REVERT_MODELS' environment variable first). Model randomly initialized by default if '--input' is not called.", metavar="path_to_input_file_state")
        parser.add_argument('--output', '-o', type=str, nargs=1, help="Should point to the folder where you want to save the model state (filename is automatically generated: 'convnet-monthday-num'). If 'path' is relative, the model will be saved in : $REVERT_MODELS/path/runs/model_name (make sure to define the '$REVERT_MODELS' environment variable first). Creates tensorboard traces in '$REVERT_LOGS/path'.", metavar="path_to_output_folder")
    
    return parser

def generate_filename(path, prefix='convnet'):
    date = datetime.datetime.now()
    day_num = str(date.day)
    month_num = str(date.month)
    datetime_object = datetime.datetime.strptime(month_num, "%m")
    month_name = datetime_object.strftime("%b").lower()
    
    num = 1
    while os.path.exists(os.path.join(path, "{}-{}{}-{}.pt".format(prefix, month_name, day_num, num))):
        num += 1
        
    return "{}-{}{}-{}".format(prefix, month_name, day_num, num)

def arg_verifier(parser, prefix='convnet'):
    args = parser.parse_args()

    if prefix == 'convnet':
        # Input
        print("--- Input ---")
        path_input = ""
        if args.input:
            path_input = args.input[0]
            if os.path.splitext(path_input)[-1].lower() != ".pt":
                raise OSError("--input should point to a '.pt' file")
            if not os.path.isabs(path_input) and 'REVERT_MODELS' in os.environ:
                path_input = os.path.join(os.environ['REVERT_MODELS'], args.input[0])
            else:
                path_input = os.path.join(os.environ['PWD'], args.input[0])
                print("Loading model from $PWD. The $REVERT_MODELS environment variable is not defined, see --help for more info.")
            print("Using {} as input".format(path_input))
            if not os.path.exists(path_input):
                raise OSError("The file {} doesn't exist".format(path_input))
        else:
            print("Model randomly initialized")

        # Output
        print("--- Output ---")
        if args.output and os.path.isabs(args.output[0]):
            path_output = args.output[0]
            path_traces = args.output[0]
        else:
            if 'REVERT_MODELS' in os.environ:
                path_output = os.environ['REVERT_MODELS']
            else:
                path_output = os.environ['PWD']
                print("Saving model from $PWD. The $REVERT_MODELS environment variable is not defined, see --help for more info.")
            if 'REVERT_LOGS' in os.environ:
                path_traces = os.environ['REVERT_LOGS']
            else:
                path_traces = os.environ['PWD']
                print("Saving model from $PWD. The $REVERT_LOGS environment variable is not defined, see --help for more info.")
            if args.output:
                path_output = os.path.join(path_output, args.output[0])
                path_traces = os.path.join(path_traces, args.output[0])
        
        file_name = generate_filename(path_output)
        path_output_file = os.path.join(path_output, file_name) + '.pt'
        path_traces = os.path.join(path_traces, 'runs', file_name)
        print("Using {} as output file".format(path_output_file))
        print("Using {} as output traces logs".format(path_traces))

        return path_input, path_output_file, path_traces
