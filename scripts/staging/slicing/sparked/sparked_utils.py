from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from slicing.base.SparkedNode import SparkedNode
from slicing.base.slicer import opt_fun, union

calc_loss = udf(lambda target, prediction, type: calc_loss_fun(target, prediction, type), FloatType())
model_type_init = udf(lambda type: init_model_type(type))


def calc_loss_fun(target, prediction, type):
    if type == 0:
        return (prediction - target) ** 2
    elif type == 1:
        if target == prediction:
            return float(1)
        else:
            return float(0)


def init_model_type(model_type):
    if model_type == "regression":
        return 0
    elif model_type == "classification":
        return 1


def slice_join_nonsense(node_i, node_j, cur_lvl):
    commons = 0
    for attr1 in node_i.attributes:
        for attr2 in node_j.attributes:
            if attr1 == attr2:
                commons = commons + 1
    return commons != cur_lvl - 1


def make_first_level(features, predictions, loss, top_k, alpha, k, w, loss_type):
    first_level = []
    # First level slices are enumerated in a "classic way" (getting data and not analyzing bounds
    for feature in features:
        new_node = SparkedNode(loss, predictions)
        new_node.parents = [feature]
        new_node.attributes.append(feature)
        new_node.name = new_node.make_name()
        new_node.key = new_node.make_key()
        new_node.process_slice(loss_type)
        new_node.score = opt_fun(new_node.loss, new_node.size, loss, len(predictions), w)
        new_node.c_upper = new_node.score
        first_level.append(new_node)
        new_node.print_debug(top_k, 0)
        # constraints for 1st level nodes to be problematic candidates
        #if new_node.check_constraint(top_k, len(predictions), alpha):
            # this method updates top k slices if needed
            #top_k.add_new_top_slice(new_node)
    return first_level


def slice_union_nonsense(node_i, node_j):
    flag = False
    for attr1 in node_i.attributes:
        for attr2 in node_j.attributes:
            if attr1 == attr2:
                # there are common attributes which is not the case we need
                flag = True
                break
    return flag


def process_node(node_i, level, loss, predictions, cur_lvl, top_k, alpha, loss_type, w, debug, enumerator):
    cur_enum_nodes = []
    for node_j in level:
        if enumerator == "join":
            flag = slice_join_nonsense(node_i, node_j, cur_lvl)
        else:
            flag = slice_union_nonsense(node_i, node_j)
        if not flag and int(node_i.name.split("&&")[0]) < int(node_j.name.split("&&")[0]):
            new_node = SparkedNode(loss, predictions)
            parents_set = set(new_node.parents)
            parents_set.add(node_i)
            parents_set.add(node_j)
            new_node.parents = list(parents_set)
            parent1_attr = node_i.attributes
            parent2_attr = node_j.attributes
            new_node_attr = union(parent1_attr, parent2_attr)
            new_node.attributes = new_node_attr
            new_node.name = new_node.make_name()
            new_node.key = new_node.make_key()
            new_node.calc_bounds(cur_lvl, w)
            to_slice = new_node.check_bounds(top_k, len(predictions), alpha)
            if to_slice:
                new_node.process_slice(loss_type)
                new_node.score = opt_fun(new_node.loss, new_node.size, loss, len(predictions), w)
                if new_node.check_bounds(top_k, len(predictions), alpha):
                    cur_enum_nodes.append(new_node)
            if debug:
                new_node.print_debug(top_k, cur_lvl)
    return cur_enum_nodes


def nodes_enum(nodes, level, predictions, loss, top_k, alpha, k, w, loss_type, cur_lvl, debug, enumerator):
    cur_enum_nodes = []
    for node_i in nodes:
        partial_nodes = process_node(node_i, level, loss, predictions, cur_lvl, top_k, alpha,
                                     loss_type, w, debug, enumerator)
        cur_enum_nodes.append(partial_nodes)
    return cur_enum_nodes


def init_top_k(first_level, top_k, alpha, predictions):
    # driver updates topK
    for sliced in first_level:
        if sliced[1].check_constraint(top_k, len(predictions), alpha):
            # this method updates top k slices if needed
            top_k.add_new_top_slice(sliced[1])


def update_top_k(new_slices, top_k, alpha, predictions):
    # driver updates topK
    for sliced in new_slices.values():
        if sliced.check_constraint(top_k, len(predictions), alpha):
            # this method updates top k slices if needed
            top_k.add_new_top_slice(sliced)
