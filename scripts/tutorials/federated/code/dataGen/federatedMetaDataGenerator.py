import argparse
import io
import json
import math

data_dir = "data/"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--ports", type=str, nargs="+", required=True)
parser.add_argument("-a", "--addresses", type=str, nargs="+", required=True)
parser.add_argument("-n", "--numberOfWorkers", required=True, type=int)
parser.add_argument("-d", "--dataName", required=True, type=str)
parser.add_argument("-f", "--features", required=True, type=int)
parser.add_argument("-e", "--num_examples", required=True, type=int)
args = parser.parse_args()

data_name = args.dataName
addresses = args.addresses
features = args.features
ports = args.ports
num_workers = args.numberOfWorkers
num_examples = args.num_examples

examples_per_worker = math.ceil(num_examples / num_workers)


def writeMtd(rows: int, cols: int, name: str):
    features_mtd_mtd = {
        "data_type": "matrix",
        "value_type": "double",
        "rows": rows,
        "cols": cols,
        "nnz": -1,
        "format": "federated",
    }
    with io.open(data_dir + name, "w", encoding="utf-8") as f:
        f.write(json.dumps(features_mtd_mtd, ensure_ascii=False, indent=4))


def writeJson(rows: int, cols: int, name: str, shortName: str):
    data = []
    if num_workers == 1:
        data.append(
            {
                "dataType": "MATRIX",
                "address": "localhost:" + ports[0],
                "filepath": data_dir + shortName[:-1] + ".data",
                "begin": [0, 0],
                "end": [rows, cols],
            }
        )
    else:
        for i in range(0, num_workers):
            start = examples_per_worker * i
            end = min(examples_per_worker * (i + 1), rows)

            data.append(
                {
                    "dataType": "MATRIX",
                    "address": "localhost:" + ports[i],
                    "filepath": data_dir
                    + shortName
                    + str(num_workers)
                    + "_"
                    + str(i + 1)
                    + ".data",
                    "begin": [start, 0],
                    "end": [end, cols],
                }
            )

    with io.open(data_dir + name, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


writeJson(
    num_examples, features, "fed_" + data_name + str(num_workers) + ".json", data_name
)

writeMtd(num_examples, features, "fed_" + data_name + str(num_workers) + ".json.mtd")
