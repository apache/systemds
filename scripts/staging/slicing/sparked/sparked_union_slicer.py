from pyspark import SparkContext

from slicing.base.top_k import Topk
from slicing.sparked import sparked_utils
from slicing.sparked.sparked_utils import update_top_k


def process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type, enumerator):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    all_features = list(all_features)
    first_level = {}
    first_tasks = sc.parallelize(all_features)
    SparkContext.broadcast(sc, top_k)
    init_slices = first_tasks.mapPartitions(lambda features: sparked_utils.make_first_level(features, predictions, loss, top_k,
                                                        alpha, k, w, loss_type)).map(lambda node: (node.key, node)).collect()
    first_level.update(init_slices)
    update_top_k(first_level, top_k, alpha, predictions)
    SparkContext.broadcast(sc, top_k)
    SparkContext.broadcast(sc, first_level)
    levels.append(first_level)
    cur_lvl = 1
    top_k.print_topk()
    SparkContext.broadcast(sc, top_k)
    while len(levels[cur_lvl - 1]) > 0:
        cur_lvl_res = {}
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - left - 1
            partitions = sc.parallelize(levels[left].values())
            mapped = partitions.mapPartitions(lambda nodes: sparked_utils.nodes_enum(nodes, levels[right].values(), predictions,loss,
                                                                      top_k, alpha, k, w, loss_type, cur_lvl, debug, enumerator))
            flattened = mapped.flatMap(lambda node: node)
            partial = flattened.map(lambda node: (node.key, node)).collect()
            cur_lvl_res.update(partial)
        SparkContext.broadcast(sc, cur_lvl_res)
        levels.append(cur_lvl_res)
        SparkContext.broadcast(sc, top_k)
        update_top_k(cur_lvl_res, top_k, alpha, predictions)
        SparkContext.broadcast(sc, top_k)
        cur_lvl = cur_lvl + 1
        top_k.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(levels) * (len(levels) - 1)) +
              " candidates but after pruning only " + str(len(cur_lvl_res)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl))
    print()
    print("Selected slices are: ")
    top_k.print_topk()


