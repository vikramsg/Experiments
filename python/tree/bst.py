
class Node:
    def __init__(self, data):
        self.data  = data 
        self.left  = None
        self.right = None

class Tree:
    """
    Let us first create a tree for 2x2 grid 
    """
    def insert(self, root, data):
        if (root == None):
            return Node(data)

        if (data <= root.data):
            root.left = self.insert(root.left, data)
        else:
            root.right = self.insert(root.right, data)

        return root


    def breadthPrint(self, root):
        if (root == None):
            print("Empty tree")
            return
        print("Here comes the tree")
        print(root.data)
        lst = [root.left, root.right]
        while True:
            lstNew = []
            s   = ""
            ctr = 0
            for i in lst:
                if (i is None):
                    s = s + "None "
                else:
                    ctr = 1
                    s = s + str(i.data) + " " 
                    lstNew.extend([i.left, i.right])

            lst = lstNew

            print("Next level")
            print(s)
            if (ctr == 0):
                break



if __name__=="__main__":

    ob = Tree()

    root = None

    root = ob.insert(root, 2)
    root = ob.insert(root, 4)
    root = ob.insert(root, 3)

    ob.breadthPrint(root)
