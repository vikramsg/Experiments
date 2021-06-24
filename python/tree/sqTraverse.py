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
    def __init__(self):
        self.rowSize = 2

        self.size    = self.rowSize*self.rowSize

    
    def getIndices(self, data):
        """
        Find valid left, right, top, bot indices otherwise return None
        In a given row, we take the mod so that it only goes from
        0:rowSize - 1. Then we check if +1, -1 lie within 0:rowSize
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


    def startInsert(self, root, data):
        ## An array that lists if a given point has been traversed
        traversed = []
        for i in range(self.size):
            traversed.append(0)

        self.insert(root, traversed, data)

#        [indLeft, indRight, indTop, indBot] = self.getIndices(data)
#        print( self.getIndices(data) )

        return root


if __name__=="__main__":

    ob = SqrTree()

    root = None

    ob.startInsert(root, 0)








