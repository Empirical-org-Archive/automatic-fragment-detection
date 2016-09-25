import re

class Node:
    def __init__(self):
        self.label = ''
        self.parent = -100
        self.parent_n = None
        self.elder_siblings = []
        self.younger_siblings = []
        self.children = []
        self.word = ''
        self.word_index = -100
        self.id = ''

    def search_self_and_younger_siblings(self, l):
        out = None
        if self.label in l:
            out = self
        else:
            for x in self.younger_siblings:
                if x.label in l:
                    out = x
                    break
        return out

    def search_younger_siblings(self, l):
        out = None
        for x in self.younger_siblings:
            if x.label in l:
                out = x
                break
        return out

    def search_children(self, l):
        out = None
        for x in self.children:
            if x.label in l:
                out = x
                break
        return out

    def match_children(self, l, find_last = False):
        out = None
        for x in self.children:
            if l in x.label:
                out = x
                if not find_last:
                    break
        return out

    def check_and_get_child(self, index, l):
        out = None
        if len(self.children) > index and self.children[index].label in l:
            out = self.children[index]
        return out

    def search_children_recursive(self, l):
        out = self.search_children(l)
        if out == None:
            for x in self.children:
                temp = x.search_children_recursive(l)
                if temp != None:
                    out = temp
                    break
        return out

    def match_children_recursive(self, l, find_last = False):
        out = self.match_children(l, find_last)
        if out == None:
            for x in self.children:
                temp = x.match_children_recursive(l)
                if temp != None:
                    out = temp
                    if not find_last:
                        break
        return out


def search_tree(tree, word_list):
    target = word_list[:]
    retrieve = False
    out = []
    words = []
    level = [-1,]
    label = ''
    is_label = False
    is_word = True
    word = ''
    output = ''
    output2 = -123
    for x in tree:
        if x == '(':
            level.append(len(out))
            is_label = True
            is_word = False
            word = ''
        elif x == ')':
            is_word = False
            if word.strip() != '':
                words.append(word.strip())
                if len(target) > 0 and word == target[0]:
                    target.pop(0)
                    if len(target) == 0:
                        output2 = level[-1]
                        retrieve = True
            word = ''
            temp = level.pop()
            out[temp][2] = len(words)
        elif is_label:
            if x == ' ':
                is_label = False
                out.append([label, [], 0, len(out)])
                if retrieve:
                    output = label
                    retrieve = False
                if level[-2] >= 0:
                    out[level[-2]][1].append(len(out)-1)
                label = ''
                is_word = True
            else:
                label += x
        elif is_word:
            word += x
    return output2

def search_label(tree, word_list):
    target = word_list[:]
    retrieve = False
    out = []
    words = []
    level = [-1,]
    label = ''
    is_label = False
    is_word = True
    word = ''
    output = ''
    for x in tree:
        if x == '(':
            level.append(len(out))
            is_label = True
            is_word = False
            word = ''
        elif x == ')':
            is_word = False
            if word.strip() != '':
                words.append(word.strip())
                if len(target) > 0 and word == target[0]:
                    target.pop(0)
                    if len(target) == 0:
                        output = out[level[-1]][0]
                        break
            word = ''
            temp = level.pop()
            out[temp][2] = len(words)
        elif is_label:
            if x == ' ':
                is_label = False
                out.append([label, [], 0, len(out)])
                if retrieve:
                    output = label
                    retrieve = False
                    #break
                if level[-2] >= 0:
                    out[level[-2]][1].append(len(out)-1)
                label = ''
                is_word = True
            else:
                label += x
        elif is_word:
            word += x
    return output


def parse_tree_with_words(tree):
    out = []
    words = []
    level = [-1,]
    label = ''
    is_label = False
    is_word = True
    word = ''
    for x in tree:
        if x == '(':
            level.append(len(out))
            is_label = True
            is_word = False
            word = ''
        elif x == ')':
            is_word = False
            if word.strip() != '':
                words.append(word.strip())
            word = ''
            temp = level.pop()
            out[temp][2] = len(words)
        elif is_label:
            if x == ' ':
                is_label = False
                out.append([label, [], 0])
                if level[-2] >= 0:
                    out[level[-2]][1].append(len(out)-1)
                label = ''
                is_word = True
            else:
                label += x
        elif is_word:
            word += x
    return (out, words)


def parse_tree(tree):
    temp = parse_tree_with_words(tree)
    return convert_tree(temp[0], temp[1])

def convert_tree(tree, words):
    nodes = []
    temp_words = words[:]
    word_indexes = list(range(1, len(words)+1))
    for x in tree:
        nodes.append(Node())
    for i, x in enumerate(tree):
        nodes[i].id = i
        nodes[i].label = x[0]
        nodes[i].word_index = x[2]
        nodes[i].children = [nodes[k] for k in x[1]]
        if len(word_indexes) > 0 and x[2] == word_indexes[0] and len(x[1]) == 0:
            word_indexes.pop(0)
            nodes[i].word = temp_words.pop(0)
        for j, y in enumerate(x[1]):
            nodes[y].parent = i
            nodes[y].parent_n = nodes[i]
            nodes[y].elder_siblings = [nodes[k] for k in x[1][:j]]
            nodes[y].younger_siblings = [nodes[k] for k in x[1][j+1:]]
    return nodes

def tree_sub_func(matchobj):
    return "|" + matchobj.group(0)

def get_trees_from_raw(trees_raw, flat=True):
    trees = trees_raw
    if flat:
        trees = trees.replace('\n', '')
    trees = re.sub(r"\(ROOT", tree_sub_func, trees)
    trees = trees.split('|')[1:]
    return trees
