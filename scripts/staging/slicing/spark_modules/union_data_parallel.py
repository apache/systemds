from pyspark import SparkContext

from slicing.base.Bucket import Bucket
from slicing.base.SparkNode import SparkNode
from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils, join_data_parallel
from slicing.spark_modules.join_data_parallel import rows_mapper, combiner
from slicing.spark_modules.spark_utils import approved_union_slice


def merge_values(a, b):
    a.minimize_bounds(b)
    return a


def merge_combiners(a, b):
    a + b
    return a


def union_enum(left_level, right_level, x_size, alpha, top_k, w, loss, cur_lvl):
    buckets = {}
    for node_i in range(len(left_level)):
        for node_j in range(len(right_level)):
            flag = approved_union_slice(left_level[node_i], right_level[node_j])
            if flag:
                node = SparkNode(None, None)
                node.attributes = list(set(left_level[node_i].attributes) | set(right_level[node_j].attributes))
                bucket = Bucket(node, cur_lvl, w, x_size, loss)
                bucket.parents.append(left_level[node_i])
                bucket.parents.append(right_level[node_j])
                bucket.calc_bounds(w, x_size, loss)
                if bucket.check_bounds(x_size, alpha, top_k):
                    buckets[bucket.name] = bucket
    return buckets


def parallel_process(all_features, predictions, loss, sc, debug, alpha, k, w, loss_type):
    top_k = Topk(k)
    cur_lvl = 0
    levels = []
    cur_lvl_nodes = list(all_features)
    pred_pandas = predictions.toPandas()
    x_size = len(pred_pandas)
    b_topk = SparkContext.broadcast(sc, top_k)
    b_cur_lvl = SparkContext.broadcast(sc, cur_lvl)
    buckets = {}
    for node in cur_lvl_nodes:
        bucket = Bucket(node, b_cur_lvl.value, w, x_size, loss)
        buckets[bucket.name] = bucket
    b_buckets = SparkContext.broadcast(sc, buckets)
    rows = predictions.rdd.map(lambda row: (row[0], row[1].indices, row[2])) \
        .map(lambda item: (item[0], item[1].tolist(), item[2]))
    mapped = rows.map(lambda row: rows_mapper(row, b_buckets.value, loss_type))
    flattened = mapped.flatMap(lambda line: (line.items()))
    reduced = flattened.combineByKey(combiner, join_data_parallel.merge_values, join_data_parallel.merge_combiners)
    cur_lvl_nodes = reduced.values() \
        .map(lambda bucket: spark_utils.calc_bucket_metrics(bucket, loss, w, x_size, b_cur_lvl.value))
    if debug:
        cur_lvl_nodes.map(lambda bucket: bucket.print_debug(b_topk.value)).collect()
    cur_lvl = 1
    prev_level = cur_lvl_nodes.collect()
    b_cur_lvl_nodes = SparkContext.broadcast(sc, prev_level)
    levels.append(b_cur_lvl_nodes)
    top_k = b_topk.value.buckets_top_k(prev_level, x_size, alpha)
    while len(levels[cur_lvl - 1].value) > 0:
        b_topk = SparkContext.broadcast(sc, top_k)
        b_cur_lvl = SparkContext.broadcast(sc, cur_lvl)
        b_topk.value.print_topk()
        buckets = []
        for left in range(int(cur_lvl / 2) + 1):
            right = cur_lvl - left - 1
            nodes = union_enum(levels[left].value, levels[right].value, x_size, alpha, b_topk.value, w, loss, b_cur_lvl.value)
            buckets.append(nodes)
        b_buckets = sc.parallelize(buckets)
        all_buckets = b_buckets.flatMap(lambda line: (line.items()))
        combined = dict(all_buckets.combineByKey(combiner, merge_values, merge_combiners).collect())
        b_buckets = SparkContext.broadcast(sc, combined)
        to_slice = dict(filter(lambda bucket: bucket[1].check_bounds(x_size, alpha, b_topk.value), b_buckets.value.items()))
        mapped = rows.map(lambda row: rows_mapper(row, to_slice, loss_type))
        flattened = mapped.flatMap(lambda line: (line.items()))
        partial = flattened.combineByKey(combiner, join_data_parallel.merge_values, join_data_parallel.merge_combiners)
        prev_level = partial\
            .map(lambda bucket: spark_utils.calc_bucket_metrics(bucket[1], loss, w, x_size, b_cur_lvl.value)).collect()
        if debug:
            partial.values().map(lambda bucket: bucket.print_debug(b_topk.value)).collect()
        top_k = b_topk.value.buckets_top_k(prev_level, x_size, alpha)
        print("Level " + str(cur_lvl) + " had " + str(
            len(levels[cur_lvl - 1].value) * (len(levels[cur_lvl - 1].value) - 1)) +
              " candidates but after pruning only " + str(len(prev_level)) + " go to the next level")
        print("Program stopped at level " + str(cur_lvl))
        b_cur_lvl_nodes = SparkContext.broadcast(sc, prev_level)
        levels.append(b_cur_lvl_nodes)
        cur_lvl = b_cur_lvl.value + 1
    print()
    print("Selected slices are: ")
    b_topk.value.print_topk()
    return None
