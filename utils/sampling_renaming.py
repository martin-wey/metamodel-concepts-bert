import json

from pathlib import Path


def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)

ROOT_TAG = '<MODEL>'
CLS_TAG = '<CLS>'
CLS_NAME_TAG = '<NAME>'
ATTRS_TAG = '<ATTRS>'
ASSOCS_TAG = '<ASSOCS>'


class Tree:
    def __init__(self, open_char='(', close_char=')'):
        self.open_char = open_char
        self.close_char = close_char
        self.tree_string = None

    def generate_local_context_samples(self, data):
        for idx, cls in enumerate(data['children']):
            current_cls = cls['name']
            linked_classifiers = []
            # retrieve the classifiers with which the current
            #   one has an incoming association
            if len(cls['assocs']) > 0:
                for assocs in cls['assocs']:
                    linked_cls = [key for key, value in assocs.items()][0]
                    linked_classifiers.append(linked_cls)
    
            # retrieve the classifiers that have an incoming
            #   association to the current one
            for idx2, cls2 in enumerate(data['children']):
                cls_name = cls2['name']
                if len(cls2['assocs']) > 0:
                    for assocs2 in cls2['assocs']:
                        linked_cls2 = [key for key, value in assocs2.items()][0]
                        if linked_cls2 == current_cls and cls_name not in linked_classifiers:
                            linked_classifiers.append(cls_name)

            # order the linked_classifiers by idx
            ordered_linked_classifiers = {}
            for idx2, cls2 in enumerate(data['children']):
                if cls2['name'] == cls['name']:
                    ordered_linked_classifiers[str(idx2)] = cls['name']
                if cls2['name'] in linked_classifiers:
                    ordered_linked_classifiers[str(idx2)] = cls2['name']

            if len(ordered_linked_classifiers) > 1:
                tree_string = self.open_char + ' '
                tree_string += ROOT_TAG
                for cls in data['children']:
                    if cls['name'] in linked_classifiers or cls['name'] == current_cls:
                        tree_string += ' ' + self.open_char + ' ' + CLS_TAG + ' '
                        tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name'] + ' ' + self.close_char
                        if len(cls['attrs']) > 0:
                            tree_string += ' ' + self.open_char + ' ' + ATTRS_TAG
                            for attrs in cls['attrs']:
                                for key, value in attrs.items():
                                    tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                            tree_string += ' ' + self.close_char
                        if len(cls['assocs']) > 0:
                            tree_string += ' ' + self.open_char + ' ' + ASSOCS_TAG
                            for assocs in cls['assocs']:
                                for key, value in assocs.items():
                                    tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                            tree_string += ' ' + self.close_char
                        tree_string += ' ' + self.close_char
                tree_string += ' ' + self.close_char + ';' + current_cls
                yield tree_string

    def generate_cls_samples(self, data):
        tree_string = self.open_char + ' '
        tree_string += ROOT_TAG

        i = 0
        for idx, cls in enumerate(data['children']):
            tree_string += ' ' + self.open_char + ' ' + CLS_TAG + ' '
            if 0 < i == idx:
                tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name']
                yield tree_string
            else:
                tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name'] + ' ' + self.close_char

            if len(cls['attrs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ATTRS_TAG
                for attrs in cls['attrs']:
                    for key, value in attrs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char
            if len(cls['assocs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ASSOCS_TAG
                for assocs in cls['assocs']:
                    for key, value in assocs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char
            tree_string += ' ' + self.close_char
            i += 1

    def generate_attrs_samples(self, data):
        tree_string = self.open_char + ' '
        tree_string += ROOT_TAG

        for idx, cls in enumerate(data['children']):
            tree_string += ' ' + self.open_char + ' ' + CLS_TAG + ' '
            tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name'] + ' ' + self.close_char

            if len(cls['attrs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ATTRS_TAG
                j = 0
                for idx, attrs in enumerate(cls['attrs']):
                    for key, value in attrs.items():
                        if j == idx:
                            tree_string += ' ' + self.open_char + ' ' + key + ' ' + value
                            yield tree_string
                        else:
                            tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                        j += 1
                tree_string += ' ' + self.close_char

            if len(cls['assocs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ASSOCS_TAG
                for assocs in cls['assocs']:
                    for key, value in assocs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char
            tree_string += ' ' + self.close_char

    def generate_assocs_samples(self, data):
        tree_string = self.open_char + ' '
        tree_string += ROOT_TAG

        for idx, cls in enumerate(data['children']):
            tree_string += ' ' + self.open_char + ' ' + CLS_TAG + ' '
            tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name'] + ' ' + self.close_char

            if len(cls['attrs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ATTRS_TAG
                for attrs in cls['attrs']:
                    for key, value in attrs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char

            if len(cls['assocs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ASSOCS_TAG
                j = 0
                for idx, assocs in enumerate(cls['assocs']):
                    for key, value in assocs.items():
                        if j == idx:
                            tree_string += ' ' + self.open_char + ' ' + key + ' ' + value
                            yield tree_string
                        else:
                            tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                        j += 1
                tree_string += ' ' + self.close_char
            tree_string += ' ' + self.close_char

    def build_tree_string_from_dict(self, data):
        """
        Generate the complete tree structure.
        """
        tree_string = self.open_char + ' '
        tree_string += ROOT_TAG
        for cls in data['children']:
            tree_string += ' ' + self.open_char + ' ' + CLS_TAG + ' '
            tree_string += self.open_char + ' ' + CLS_NAME_TAG + ' ' + cls['name'] + ' ' + self.close_char
            if len(cls['attrs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ATTRS_TAG
                for attrs in cls['attrs']:
                    for key, value in attrs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char
            if len(cls['assocs']) > 0:
                tree_string += ' ' + self.open_char + ' ' + ASSOCS_TAG
                for assocs in cls['assocs']:
                    for key, value in assocs.items():
                        tree_string += ' ' + self.open_char + ' ' + key + ' ' + value + ' ' + self.close_char
                tree_string += ' ' + self.close_char
            tree_string += ' ' + self.close_char
        tree_string += ' ' + self.close_char
        self.tree_string = tree_string
        return self.tree_string

    def tree_to_file(self, path):
        with open(path, 'w+') as f:
            f.write(self.tree_string)


"""
path = '../data/test/use_case3'

with open('../data/test/use_case3/test_probing_local_context.txt', 'w+') as fout:
    for path in Path(path).rglob('*.json'):
        if path.is_file():
            tree = Tree()
            test_samples = tree.generate_contiguous_samples(load_json_file(path))
            for sample in test_samples:
                fout.write(sample)
                fout.write('\n')
"""