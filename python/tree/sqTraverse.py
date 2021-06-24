import random
import sys

#sys.setrecursionlimit(10)

class Node:
    def __init__(self, data):
        """
        We can move in only 4 directions
        """
        self.data  = data 
        self.left  = None
        self.right = None
        self.top   = None
        self.bot   = None


class SqrTree:
    """
    Creating a tree for a 2x2 square numbered 0, 1, 2, 3 
    """
    def __init__(self, edgeSize):
        self.rowSize = edgeSize 

        self.size    = self.rowSize*self.rowSize

        self.cnt     = 0

    
    def getIndices(self, data):
        """
        Find valid left, right, top, bot indices otherwise return None
        In a given row, we take the mod so that it only goes from
        0:rowSize - 1. Then we check if +1, -1 lie within 0:rowSize - 1
        """
        tmp = data%self.rowSize - 1
        ind = data - 1
        indLeft  = ind if (( tmp < self.rowSize) and  ( tmp >= 0)) else None  

        tmp = data%self.rowSize + 1
        ind = data + 1
        indRight = ind if (( tmp < self.rowSize) and  ( tmp >= 0)) else None  

        ind = data - self.rowSize
        indTop   = ind if ( ( ind < self.size) and  ( ind >= 0) ) else None  

        ind = data + self.rowSize
        indBot   = ind if ( ( ind < self.size) and  ( ind >= 0) ) else None  

        return [indLeft, indRight, indTop, indBot]

    def insert(self, root, traversed, data):
        if (traversed[data] == 1):
            return [root, traversed]
        if (root is None):
            root = Node(data)

        trv = list(traversed) #Mimic pass by value
        trv[root.data] = 1
        print("Data: ", data)
        print("Traversed: ", trv)

        traverseInd = self.getIndices(root.data)
        [indLeft, indRight, indTop, indBot] = self.getIndices(root.data)
        print("Traverse indices: ", traverseInd)

        if ( ( indLeft is not None ) ):
            root.left = self.insert(root.left, trv, indLeft)
        if ( ( indRight is not None ) ):
            root.right = self.insert(root.right, trv, indRight)
        if ( ( indTop is not None ) ):
            root.top = self.insert(root.top, trv, indTop)
        if ( ( indBot is not None ) ):
            root.bot = self.insert(root.bot, trv, indBot)

        return root

    def checkInsert(self, root, traversed, data, tr_val):
        """
        The algorithm is simple. Everytime we encounter a node
        we increement traversed[node] to the traversal count.
        This is done recursively. We attempt to go to the neigbouring index
        using getIndices, if it has not already been traversed. 
        If not we add those as leaves to the present level, which constructs
        a tree. Finally, if we come reach the final level, then we 
        increment the global counter

        tr_val keeps count of the traversal, that is, if the present 
        data was encountered on the 4th go, then tr_val is 4
        """
        if (traversed[data] > 0):
            return root
        if (root is None):
            root = Node(data)

        trv = list(traversed) #Mimic pass by value
        tr_val = tr_val + 1
        trv[root.data] = tr_val
        chk = 1
        for i in trv:
            if (i == 0):
                chk = 0
        org = []
        for i in range(self.size):
            org.append(i)
        if (chk == 1):
            self.cnt = self.cnt + 1
            dct = list(org)
            for it, i in enumerate(trv):
                dct[i - 1] = it
            print("Traversed:        ", dct)
            print("Curve count:      ", self.cnt)
            print("")

        traverseInd = self.getIndices(root.data)
        [indLeft, indRight, indTop, indBot] = self.getIndices(root.data)

        if ( ( indLeft is not None ) ):
            root.left = self.checkInsert(root.left, trv, indLeft, tr_val)
        if ( ( indRight is not None ) ):
            root.right = self.checkInsert(root.right, trv, indRight, tr_val)
        if ( ( indTop is not None ) ):
            root.top = self.checkInsert(root.top, trv, indTop, tr_val)
        if ( ( indBot is not None ) ):
            root.bot = self.checkInsert(root.bot, trv, indBot, tr_val)

        return root

    def startInsert(self, root, data):
        ## An array that lists if a given point has been traversed
        traversed = []
        for i in range(self.size):
            traversed.append(0)

        cnt    = 0
        tr_val = 0
        self.checkInsert(root, traversed, data, tr_val)
#        self.insert(root, traversed, data)

#        [indLeft, indRight, indTop, indBot] = self.getIndices(data)
#        print( self.getIndices(data) )

        return root


if __name__=="__main__":

    ob = SqrTree(3)

    root = None

    ob.startInsert(root, 4)








