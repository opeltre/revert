import argparse
import datetime
import os

def arg_parser(prefix='convnet'):
    parser = argparse.ArgumentParser()

    if prefix == 'convnet':
        env_variables = ['REVERT_MODELS', 'REVERT_LOGS']
        for var in env_variables:
            if not var in os.environ:
                raise OSError("The {} environment variable is not defined.".format(var))
        
        parser.add_argument('--input', '-i', type=str, nargs=1, help="input state name => '$REVERT_MODELS/{in}.pt' (model randomly initialized otherwise)", metavar="file_name_without_extension")
        parser.add_argument('--output', '-o', type=str, nargs=1, help="output state name => '$REVERT_MODELS/{out}.pt' and creates tensorboard traces in '$REVERT_LOGS/{out}' (out = 'convnet-apr28-1' by default for instance)", metavar="file_name_without_extension")
    
    return parser

def generate_filename(prefix='convnet', models_dir='REVERT_MODELS'):
    file_name = ""

    if prefix == 'convnet':
        date = datetime.datetime.now()
        day_num = str(date.day)
        month_num = str(date.month)
        datetime_object = datetime.datetime.strptime(month_num, "%m")
        month_name = datetime_object.strftime("%b").lower()
        
        num = 1
        while os.path.exists("{}/convnet-{}{}-{}.pt".format(os.environ[models_dir], month_name, day_num, num)):
            num += 1
            
        file_name = "{}/convnet-{}{}-{}.pt".format(os.environ[models_dir], month_name, day_num, num)
    
    return file_name

def arg_verifier(parser, prefix='convnet'):
    args = parser.parse_args()

    if prefix == 'convnet':
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
            file_name = generate_filename()
            print("Using default output: {}".format(file_name))
