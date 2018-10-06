import argparse
import yaml

""" A module for parsing arguments from config files and the command
line in YAML. The command line arguments are simply appended at the end
of what is read from a config file, which makes them take priority.
"""

def parse_argv(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configfile",
        metavar="EXAMPLE.yaml",
        help="Path to the configuration file, which should be valid YAML 1.1.",
    )
    parser.add_argument(
        "-y", "--yaml",
        metavar="STR1 STR2 ...",
        help=("Additional lines of YAML to be appended to the config file."
              " Can be supplied even if no config file is given."),
        nargs="*"
    )

    args = parser.parse_args(argv[1:])
    yamlstr = ""
    if args.configfile:
        with open(args.configfile) as f:
            yamlstr += f.read()
    if args.yaml:
        for line in args.yaml:
            yamlstr += "\n{}".format(line)
    pars = yaml.load(yamlstr)
    if pars is None:
        pars = dict()
    return pars

