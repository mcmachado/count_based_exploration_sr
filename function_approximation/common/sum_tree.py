import numpy as np

class SumTree:
    def __init__(self, size, dtype=np.float):
        '''
        Initializes a full binary tree with `size` number of leaf nodes.
        All intermediate nodes in the tree represent the sum of all nodes beneath it.
        We can use this data structure to efficiently sample from a probability distribution
        where leaf nodes represent a certain probability mass. See @SumTree.find

        Arguments:
            size: number of leaf nodes
        '''
        self._max_size = size

        self._height = (np.ceil(np.log2(size + 1)) + 1).astype(np.int32)
        self._tree = np.zeros(2 ** self._height - 1, dtype=dtype)
        self._index = 0
    
    @property
    def _leaf_index(self):
        '''
        current leaf index corresponding to self._index
        '''
        return (2 ** (self._height - 1) - 1 + self._index).astype(np.int32)

    @property
    def root(self):
        '''
        root node of the tree corresponding to the cumulative sum of all leaf nodes
        '''
        return self._tree[0]

    def __getitem__(self, index):
        '''
        Get a specified leaf nodes value at `index`
        '''
        if isinstance(index, int):
            if np.abs(index) >= self._max_size:
                raise KeyError("Index out of bounds")
        return self._tree[2 ** (self._height - 1) - 1 + index]

    def append(self, value):
        '''
        Appends value `value` to the next leaf node.
        Wrapping around if the size exceeds maximum size. This will
        update all intermediate nodes in the tree with the new sum
        from all children nodes.
        
        Also stores the data value corresponding with that leaf node.

        Uses array based representation for full binary tree. That is:
            left child = 2i + 1
            right child = 2i + 2
            parent = (i-1)/2
        '''
        self._bubble_up(self._leaf_index, value - self._tree[self._leaf_index])
        self._index = (self._index + 1) % self._max_size

    def find(self, value):
        '''
        Find the highest index `i` such that sum(leaf node from 0 .. i - 1) <= value.

        We can use this to sample a leaf node given the probability mass represented by
        the value of all leaf nodes.

        Arguments:
            value: should be between 0 .. sum(leaf node from 0 .. N - 1)

        Return:
            largest index `i` which satisfies the condition above
        '''
        index = 0
        # While index is not a leaf node
        while 2 ** (self._height  - 1) - 1 > index:
            # Check left branch, ideally we want right branch
            # Right branch means index is higher in the tree
            if value <= self._tree[2*index + 1]:
                index = 2*index + 1
            else:
                # Subtract sum of left subtree as we want index from 0 .. i - 1
                # left subtree's sum has been accounted already.
                value -= self._tree[2*index + 1]
                index = 2*index + 2
        leaf_index = index - (2 ** (self._height - 1) - 1)
        assert leaf_index >= 0
        return leaf_index

    def update(self, index, value):
        '''
        Update leaf index `index` with value and bubble the new value up

        Arguments:
            index: leaf node index to update
            value: numeric value to insert at leaf node and bubble up
        '''
        leaf_index = 2 ** (self._height - 1) - 1 + index
        self._bubble_up(leaf_index, value - self._tree[leaf_index])

    def _bubble_up(self, index, value):
        '''
        Bubbles the value `value` up the tree starting at `index`
        adding the value to each intermediate node in the tree.

        Invarient:
            - Each node is the sum of all nodes beneath it.

        Arguments:
            index: leaf node index to start bubbling up from
            value: value to set at leaf node and to bubble upwards
        '''
        self._tree[index] += value
        while index > 0:
            index = (index - 1) // 2
            self._tree[index] += value
