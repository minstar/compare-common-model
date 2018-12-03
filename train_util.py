import numpy as np
import queue
import math
# reference
# https://stackoverflow.com/questions/11587044/how-can-i-create-a-tree-for-huffman-encoding-and-decoding
class HuffmanNode(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    def children(self):
        return ((self.left, self.right))

class Huffman_encoding(object):
    def __init__(self, freq):
        self.freq = freq
    def __lt__(self,other):
        return 0

    def create_tree(self):
        p = queue.PriorityQueue()    # retrieve order with smallest order first
        for key, value in self.freq.items():# 1. Create a leaf node for each symbol and add it to the priority queue
            p.put((value,key,key))

        idx = len(self.freq)
        while p.qsize() > 1:         # 2. While there is more than one node
            l, r = p.get(), p.get()  # 2a. remove two highest nodes
            node = HuffmanNode(l, r) # 2b. create internal node with children
            p.put((l[0]+r[0], idx, node)) # 2c. add new node to queue
            idx += 1
        return p.get()               # 3. tree is complete - return root node

    # Recursively walk the tree down to the leaves,
    #   assigning a code value to each symbol
    def walk_tree(self, node, prefix="", code={}):
        if isinstance(node[2].left[2], HuffmanNode):
            self.walk_tree(node[2].left, prefix+"0", code)
        else:
            code[node[2].left[2]]=prefix+"0"
        if isinstance(node[2].right[2], HuffmanNode):
            self.walk_tree(node[2].right, prefix+"1", code)
        else:
            code[node[2].right[2]]=prefix+"1"
        return(code)

    def encoding(self):
        node = self.create_tree()
        code = self.walk_tree(node)
        self.code = code

    def make_node_dict(self):
        node_dict = dict()
        reverse_node_dict = dict()
        node_list = set()

        for word, value in self.code.items():
            pointer = 0
            for encode_string in value:
                if encode_string == '0':
                    node_list.add(pointer)
                    pointer = pointer * 2 + 1 # left node
                elif encode_string == '1':
                    node_list.add(pointer)
                    pointer = pointer * 2 + 2 # right node

        for idx, node in enumerate(node_list):
            node_dict[node] = idx
            reverse_node_dict[idx] = node

        return node_list, node_dict, reverse_node_dict
