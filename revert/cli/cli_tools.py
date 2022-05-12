import argparse
import datetime
import os

def arg_parser():
    """
    Return an ArgumentParser object with input, output and data arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str, metavar="path_to_input_file_state",
        help="""
        Initial model state path, should target an existing '.pt' file (default=None).

        If 'path' is relative and the '$REVERT_MODELS' environment variable
        is defined, the state file will be looked inside this folder.
        Model is randomly initialized otherwise.
        """)
    parser.add_argument('--output', '-o', type=str, metavar="path to dataset",
        help="""
        Path where final model state should be saved (default=None).

        If 'path' is relative and the '$REVERT_MODELS' environment variable
        is defined, the state file will be saved inside this folder,
        with tensorboard traces saved in '$REVERT_LOGS/path'.
        File name automatically generated if not given.
        """)
    parser.add_argument('--data', '-d', type=str, metavar="path to dataset",
        help="""
        Path to dataset.

        If 'path' is relative, will look inside '$INFUSION_DATASETS' or '$PCMRI_DATASETS'.
        """)
    return parser


def try_envdir (name):
    """ Return $name or $PWD """
    if name in os.environ and os.path.isdir(os.environ[name]):
        return os.environ[name]
    print(f"warning: ${name} is undefined, falling back to $PWD\n")
    return os.getcwd()


def generate_filename(dirname, prefix='module', use_date=True):
    """
    Generate a filename index from module prefix and date string.
    """
    # time info
    if use_date:
        date = datetime.datetime.now()
        day_num = str(date.day)
        month_num = str(date.month)
        datetime_object = datetime.datetime.strptime(month_num, "%m")
        month_name = datetime_object.strftime("%b").lower()
        daykey = f"{month_name[:3]}{day_num}"
    # prefix
    basename = f"{prefix}-{daykey}" if use_date else prefix
    # next slot from prefix
    num = 1
    while os.path.exists(os.path.join(dirname, f"{basename}-{num}.pt")):
        num += 1
    return os.path.join(dirname, f"{basename}-{num}.pt")

def read_args(parser, name='module', datatype=None, **defaults):
    """
    Return argument namespace after some processing.
    """

    args = parser.parse_args()
    models_dir = try_envdir('REVERT_MODELS')
    logs_dir   = try_envdir('REVERT_LOGS')
    #- Join with optional "dirname" prefix
    if "dirname" in defaults:
        models_dir = os.path.join(models_dir, defaults["dirname"])
        logs_dir   = os.path.join(logs_dir, defaults["dirname"])
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    for k, v in defaults.items():
        if not k in dir(args) or isinstance(getattr(args, k), type(None)):
            setattr(args, k, v)

    #--- Input
    if args.input:
        if os.path.splitext(args.input)[1] == "":
            args.input = os.path.extsep.join((args.input, "pt"))
        args.input = (args.input if os.path.isabs(args.input)
                                 else os.path.join(models_dir, args.input))
        print(f"> read initial {name} state from {args.input}")
    else:
        print(f"> randomly initialize {name}")

    #--- Output
    if args.output:
        if os.path.splitext(args.output)[1] == "":
            args.output = os.path.extsep.join((args.output, "pt"))
        args.output = (args.output if os.path.isabs(args.output)
                                 else os.path.join(models_dir, args.output))
    else:
        args.output = generate_filename(models_dir, name)
    print(f"> save final {name} state as {args.output}")

    #--- Dataset path
    if args.data and datatype and not os.path.isabs(args.data):
        envname = datatype.upper() + '_DATASETS'
        dbpath = os.environ[envname] if envname in os.environ else os.getcwd()
        args.data = os.path.join(dbpath, args.data)

    #--- Tensorboard
    key = os.path.splitext(os.path.basename(args.output))[0]
    args.writer = os.path.join(logs_dir, key)
    print(f"> save tensorboard traces as {args.writer}")

    return args

def parse_args(**defaults):
    """
    Process and return CLI arguments after applying defaults dict.
    """
    return read_args(arg_parser(), **defaults)

if __name__ == '__main__':
    args = read_args(arg_parser())
