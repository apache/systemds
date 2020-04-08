from pyspark import SparkContext

from slicing.base.top_k import Topk
from slicing.sparked import sparked_utils
from slicing.sparked.sparked_utils import update_top_k


def update_nodes(partial_res, nodes_list, cur_lvl_res, w):
    for item in partial_res:
        if item.key not in cur_lvl_res:
            cur_lvl_res[item.key] = item
            nodes_list.append(item)
        else:
            cur_lvl_res[item.key].update_bounds(item.s_upper, item.s_lower, item.e_upper, item.e_max_upper, w)
    return cur_lvl_res, nodes_list


def process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type, enumerator):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    all_features = list(all_features)
    first_tasks = sc.parallelize(all_features)
    SparkContext.broadcast(sc, top_k)
    first_level = first_tasks.mapPartitions(
        lambda features: sparked_utils.make_first_level(features, predictions, loss, top_k,
                                                        alpha, k, w, loss_type)).collect()
    update_top_k(first_level, top_k, alpha, predictions)
    SparkContext.broadcast(sc, top_k)
    SparkContext.broadcast(sc, first_level)
    levels.append(first_level)
    # SparkContext.broadcast(sc, levels)
    cur_lvl = 1
    top_k.print_topk()
    SparkContext.broadcast(sc, top_k)
    while len(levels[cur_lvl - 1]) > 0:
        cur_lvl_res = {}
        nodes_list = []
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - left - 1
            partitions = sc.parallelize(levels[left])
            mapped = partitions.mapPartitions(lambda nodes: sparked_utils.nodes_enum(nodes, levels[right], predictions,loss,
                                                                      top_k, alpha, k, w, loss_type, cur_lvl, debug, enumerator))\
                                                                                .reduce(lambda a, b: a + b)
            result = update_nodes(mapped, nodes_list, cur_lvl_res, w)
            cur_lvl_res = result[0]
            nodes_list = result[1]
        SparkContext.broadcast(sc, nodes_list)
        levels.append(nodes_list)
        # SparkContext.broadcast(sc, levels)
        SparkContext.broadcast(sc, top_k)
        update_top_k(list(nodes_list), top_k, alpha, predictions)
        SparkContext.broadcast(sc, top_k)
        cur_lvl = cur_lvl + 1
        top_k.print_topk()
        print("Level " + str(cur_lvl) + " had " + str(len(levels) * (len(levels) - 1)) +
              " candidates but after pruning only " + str(len(cur_lvl_res)) + " go to the next level")
    print("Program stopped at level " + str(cur_lvl))
    print()
    print("Selected slices are: ")
    top_k.print_topk()


