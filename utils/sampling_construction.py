import json
import logging
import sys
from pathlib import Path

from anytree import Node, RenderTree, LevelOrderIter, PreOrderIter
from anytree.search import findall, find

logger = logging.getLogger(__name__)

SPECIAL_TAGS = {
    'root_tag': '<MODEL>',
    'cls_tag': '<CLS>',
    'cls_name_tag': '<NAME>',
    'attrs_tag': '<ATTRS>',
    'assocs_tag': '<ASSOCS>',
    'open_char': '(',
    'close_char': ')',
    'mask_token': '<mask>'
}


def build_tree(data):
    """
    Build the initial tree using anytree based on json data.
    """
    root_node = Node(SPECIAL_TAGS['root_tag'])
    for idx, cls in enumerate(data['children']):
        cls_root = Node(SPECIAL_TAGS['cls_tag'], order=idx, parent=root_node)
        cls_name_tag = Node(SPECIAL_TAGS['cls_name_tag'], parent=cls_root)
        cls_name = Node(cls['name'], parent=cls_name_tag)
        if len(cls['attrs']) > 0:
            attrs_root = Node(SPECIAL_TAGS['attrs_tag'], parent=cls_root)
            for attrs in cls['attrs']:
                attr = [(key, value) for key, value in attrs.items()][0]
                attr_type = Node(attr[0], parent=attrs_root)
                attr_value = Node(attr[1], parent=attr_type)
        if len(cls['assocs']) > 0:
            assocs_root = Node(SPECIAL_TAGS['assocs_tag'], parent=cls_root)
            for assocs in cls['assocs']:
                assoc = [(key, value) for key, value in assocs.items()][0]
                assoc_target = Node(assoc[0], parent=assocs_root)
                assoc_value = Node(assoc[1], parent=assoc_target)

    return RenderTree(root_node)


def order_cls_by_incoming_edges(tree):
    """
    Mark all classifiers in the tree with a marker corresponding to
    their number of incoming edges (i.e., incoming associations).
    """
    # retrieve all classifier nodes
    all_cls = findall(tree.node, filter_=lambda node1: node1.parent is not None and
                                                       node1.parent.name == SPECIAL_TAGS['cls_name_tag'])
    # for each classifier node, retrieve its incoming edges and reorder
    #   the node according the number of incoming edges
    for cls in all_cls:
        cls_inc_edges = findall(
            tree.node, filter_=lambda node: node.parent is not None and
                                            node.parent.name == SPECIAL_TAGS['assocs_tag'] and
                                            node.name == cls.name)
        cls.parent.parent.order = len(cls_inc_edges)
    return tree


def count_cls(tree):
    """
    Get the number of classifiers in the tree.
    """
    all_cls = findall(tree.node, filter_=lambda node1: node1.parent is not None and
                                                       node1.parent.name == SPECIAL_TAGS['cls_name_tag'])
    return len(all_cls)


def get_cls_name(cls_node):
    """
    Get the name of a classifier node.
    """
    return cls_node.descendants[1].name


def find_cls(tree, cls_name):
    """
    Get the cls node that has the specified cls_name as name.
    """
    cls = find(tree.node, filter_=lambda node1: node1.parent is not None and
                                                    node1.parent.name == SPECIAL_TAGS['cls_name_tag'] and
                                                    node1.name == cls_name)
    return cls


def find_next_cls_node(tree, excluded_cls=[]):
    """
    Get the next classifier in the tree that is not in the specified list of excluded classifiers.
    The next classifier is chosen based on its marker (pick the lowest first).
    """
    next_cls_node = None
    i = 0
    found = False
    while not found:
        cls_lowest_inc_edge = [
            node for node in LevelOrderIter(tree.node, filter_=lambda n: n.name == SPECIAL_TAGS['cls_tag'])
            if node.order == i
        ]
        j = 0
        while j < len(cls_lowest_inc_edge) and not found:
            next_cls_node = cls_lowest_inc_edge[j]
            if get_cls_name(next_cls_node) not in excluded_cls:
                found = True
            j += 1
        i += 1
    return next_cls_node


def generate_test_sample(selected_nodes_dict, mask_node=None):
    """
    Generate a test sample with a masking tag on the specified node.
    """
    tree_string = SPECIAL_TAGS['open_char'] + ' ' + SPECIAL_TAGS['root_tag']
    ground_truth = None
    for key, items in selected_nodes_dict.items():
        cls_item = items[0]
        if cls_item == mask_node:
            current_item_name = SPECIAL_TAGS['mask_token']
            ground_truth = cls_item.name
        else:
            current_item_name = cls_item.name
        tree_string += ' ' + SPECIAL_TAGS['open_char'] + ' ' + cls_item.parent.parent.name + ' ' + \
                       SPECIAL_TAGS['open_char'] + ' ' + cls_item.parent.name + ' ' + \
                       current_item_name + ' ' + SPECIAL_TAGS['close_char']

        if len(items) > 1:
            current_special_tag = None
            for idx, item in enumerate(items[1:]):
                if item.parent.parent.name != current_special_tag:
                    if idx > 0: tree_string += ' ' + SPECIAL_TAGS['close_char']
                    tree_string += ' ' + SPECIAL_TAGS['open_char'] + ' ' + item.parent.parent.name
                if item == mask_node:
                    current_item_name = SPECIAL_TAGS['mask_token']
                    ground_truth = item.name
                else:
                    current_item_name = item.name
                tree_string += ' ' + SPECIAL_TAGS['open_char'] + ' ' + item.parent.name + \
                               ' ' + current_item_name + ' ' + SPECIAL_TAGS['close_char']
                current_special_tag = item.parent.parent.name
            tree_string += ' ' + SPECIAL_TAGS['close_char']
        tree_string += ' ' + SPECIAL_TAGS['close_char']
    tree_string += ' ' + SPECIAL_TAGS['close_char']

    # count number of elements in the context
    context_size = 0
    for key in selected_nodes_dict:
        context_size += len(selected_nodes_dict[key])

    # determine the element type to predict
    pred_type = None
    if mask_node is not None:
        if mask_node.parent.parent.name == SPECIAL_TAGS['cls_tag']:
            pred_type = 'cls'
        elif mask_node.parent.parent.name == SPECIAL_TAGS['attrs_tag']:
            pred_type = 'attrs'
        else:
            pred_type = 'assocs'

    return tree_string, context_size - 1, pred_type, ground_truth


def sample_generation(data):
    tree = build_tree(data)
    for pre, _, node in tree:
        logger.debug(f'{pre}{node.name}')

    # order each cls node according to its number of incoming edges
    ordered_tree = order_cls_by_incoming_edges(tree)
    # nodes (cls/attrs/assocs) already considered
    selected_nodes = {}
    # classifiers already considered in the construction of the metamodel
    excluded_cls = []
    # classifiers that can be reached in the tree
    #   -> (classifiers that have a link with one of the already considered classifiers)
    reachable_cls = {}
    # list of generated test samples
    test_samples = []

    current_cls = None
    current_cls_name = None
    # ensure that all classifiers are visited to produce a complete tree
    while len(excluded_cls) < count_cls(ordered_tree):
        if current_cls_name is not None and current_cls_name not in excluded_cls:
            excluded_cls.append(current_cls_name)
        if current_cls is None:
            # not been able to reach a next classifier --> pick the next one from whats left
            current_cls = find_next_cls_node(ordered_tree, excluded_cls=excluded_cls)
            current_cls_name = get_cls_name(current_cls)
            current_cls_node = current_cls.descendants[1]
            if current_cls_name not in selected_nodes:
                selected_nodes[current_cls_name] = [current_cls_node]
            # ensure that we have at least on classifier in the context to generate a test sample
            if len(selected_nodes) > 1:
                test_samples.append(
                    generate_test_sample(
                        selected_nodes_dict=selected_nodes,
                        mask_node=current_cls_node
                    )
                )
        else:
            # iterate through all attributes and associations of current classifier node
            for node in PreOrderIter(current_cls):
                if node.is_leaf and node.parent.parent.name == SPECIAL_TAGS['attrs_tag']:
                    if node not in selected_nodes[current_cls_name]:
                        selected_nodes[current_cls_name].append(node)
                        test_samples.append(
                            generate_test_sample(
                                selected_nodes_dict=selected_nodes,
                                mask_node=node
                            )
                        )
                elif node.is_leaf and node.parent.parent.name == SPECIAL_TAGS['assocs_tag']:
                    # get the cls linked to the current one with the current association
                    linked_cls = find_cls(ordered_tree, node.parent.name)
                    if linked_cls is not None:
                        if linked_cls.name not in selected_nodes:
                            # add the classifier in the selected nodes and produce a test sample
                            selected_nodes[linked_cls.name] = [linked_cls]
                            test_samples.append(
                                generate_test_sample(
                                    selected_nodes_dict=selected_nodes,
                                    mask_node=linked_cls
                                )
                            )

                        if node not in selected_nodes[current_cls_name]:
                            selected_nodes[current_cls_name].append(node)
                            test_samples.append(
                                generate_test_sample(
                                    selected_nodes_dict=selected_nodes,
                                    mask_node=node
                                )
                            )

                        for linked_cls_children in PreOrderIter(linked_cls.parent.parent):
                            if linked_cls_children.is_leaf and \
                                    linked_cls_children.parent.parent.name == SPECIAL_TAGS['attrs_tag'] and \
                                    linked_cls_children not in selected_nodes[linked_cls.name]:
                                selected_nodes[linked_cls.name].append(linked_cls_children)
                                test_samples.append(
                                    generate_test_sample(
                                        selected_nodes_dict=selected_nodes,
                                        mask_node=linked_cls_children
                                    )
                                )
                            elif linked_cls_children.is_leaf and \
                                    linked_cls_children.parent.parent.name == SPECIAL_TAGS['assocs_tag'] and \
                                    linked_cls_children.parent.name == current_cls_name and \
                                    linked_cls_children not in selected_nodes[linked_cls.name]:
                                # get associations linked to the current classifier
                                selected_nodes[linked_cls.name].append(linked_cls_children)
                                test_samples.append(
                                    generate_test_sample(
                                        selected_nodes_dict=selected_nodes,
                                        mask_node=linked_cls_children
                                    )
                                )

                        if linked_cls.name not in reachable_cls:
                            reachable_cls[linked_cls.name] = linked_cls.parent.parent

            found_next_cls = False
            while not found_next_cls:
                if len(reachable_cls) > 0:
                    next_key = list(reachable_cls.keys())[0]
                    current_cls = reachable_cls[next_key]
                    current_cls_name = get_cls_name(current_cls)
                    del reachable_cls[next_key]
                    if current_cls_name not in excluded_cls:
                        found_next_cls = True
                else:
                    current_cls = None
                    current_cls_name = None
                    found_next_cls = True

    # this last fake sample should correspond to the complete tree
    logger.debug(generate_test_sample(selected_nodes_dict=selected_nodes))

    return test_samples


"""
@todo: clean sampling scripts
"""

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level="INFO"
    )

    base_path = '../data/test/use_case3'
    logger.info('Generating test samples...')

    with open('../data/test/use_case3/test_iterative_construction.txt', 'w+') as fout:
        for path in Path(base_path).rglob('*.json'):
            logging.info(f'Parsing file: {path}')
            if path.is_file():
                with open(path) as fin:
                    data = json.load(fin)
                test_samples = sample_generation(data)
                for sample in test_samples:
                    fout.write(sample[0] + ';' + str(sample[1]) + ';' + sample[2] + ';' + sample[3])
                    fout.write('\n')
            fout.write('\n')
