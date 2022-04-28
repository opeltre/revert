import argparse
import os
from revert import pcmri
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--error', '-e', nargs=1, help="Copy all the exams having an error in a folder to be able to analyze them manually afterwards", metavar="PATH/TO/FOLDER/")
args = parser.parse_args()

if args.error:
    if not os.path.isdir('patients_json/'):
        raise OSError("The patients_json/ folder does not exist. Generate it with pcmri_to_tensor.py using the --json argument.")
    if os.path.isdir(args.error[0]):
        shutil.rmtree(args.error[0])
    os.makedirs(args.error[0], exist_ok=True)

db = pcmri.Dataset("full")

types = {'art > cervi': 0, 'art > cereb': 0, 'ven > cervi': 0, 'ven > cereb': 0}
types_channels = {}
for type in types:
    types_channels[type] = []

for channel in db.channels:
    type = db.channels[channel]['type']
    if type in types:
        types_channels[type].append(channel)

other_types = {'c2-c3': 0, 'aqueduc': 0}

inters = {}

for patient in db.getAll("*"):
    channels = [channel.replace("_", "-").lower() for channel in patient.channels]
    l = []
    for type in types:
        if set(types_channels[type]).isdisjoint(channels):
            types[type] += 1
            l.append(type)
    for type in other_types:
        if type not in channels:
            other_types[type] += 1
            l.append(type)

    if l != []:
        list_str = str(l)
        if list_str in inters:
            inters[list_str] += 1
        else:
            inters[list_str] = 1

        if args.error:
            os.link("patients_json/"+patient.id+".json", "../../errors/"+patient.id+".json")

print("Total missing:")
types.update(other_types)
for type, val in types.items():
    print("{}: {}".format(type, val))
print("\n--------------------\n")
print("Intersection:")
for key, value in sorted(inters.items(), key=lambda kv: kv[1], reverse=True):
    print("{}: {}".format(key, value))
