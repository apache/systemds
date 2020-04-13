from pyspark import SparkContext

from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils
from slicing.spark_modules.spark_utils import update_top_k


def process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type, enumerator):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    all_features = list(all_features)
    first_level = {}
    first_tasks = sc.parallelize(all_features)
    b_topk = SparkContext.broadcast(sc, top_k)
    init_slices = first_tasks.mapPartitions(lambda features: spark_utils.make_first_level(features, predictions, loss,
                                                                                          b_topk.value, w, loss_type)) \
        .map(lambda node: (node.key, node)) \
        .collect()
    first_level.update(init_slices)
    update_top_k(first_level, b_topk.value, alpha, predictions)
    prev_level = SparkContext.broadcast(sc, first_level)
    levels.append(prev_level)
    cur_lvl = 1
    b_topk.value.print_topk()
    while len(levels[cur_lvl - 1].value) > 0:
        cur_lvl_res = {}
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - left - 1
            partitions = sc.parallelize(levels[left].value.values())
            mapped = partitions.mapPartitions(lambda nodes: spark_utils.nodes_enum(nodes, levels[right].value.values(),
                                                                                   predictions, loss, b_topk.value, alpha, k,
                                                                                   w, loss_type, cur_lvl, debug,
                                                                                   enumerator))
            flattened = mapped.flatMap(lambda node: node)
            partial = flattened.map(lambda node: (node.key, node)).collect()
            cur_lvl_res.update(partial)
        prev_level = SparkContext.broadcast(sc, cur_lvl_res)
        levels.append(prev_level)
        update_top_k(cur_lvl_res, b_topk.value, alpha, predictions)
        cur_lvl = cur_lvl + 1
        top_k.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(levels[cur_lvl - 1].value) * (len(levels[cur_lvl - 1].value) - 1)) +
              " candidates but after pruning only " + str(len(prev_level.value)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl))
    print()
    print("Selected slices are: ")
    b_topk.value.print_topk()
