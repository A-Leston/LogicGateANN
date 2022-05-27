# Artificial neural network by ariel leston
from math import *
import random
import time


class Node:
    def __init__(self, ids):
        self.nodeId = ids[0]        # node number by index
        self.position = ids[1]      # model coords by index [layerNum, nodeNum on layer]
        self.inputs = []            # list of input nodes
        self.outputs = []           # list of output nodes
        self.a = None
        self.delta = None
        self.bias = 1

    def __str__(self):
        retString = f"node {self.nodeId} at position {self.position}"
        retString += f" with an outValue of {self.a}"
        return retString


class Model:
    def __init__(self, nodeMap, baseWeight=None):
        self.nodes = []
        self.layers = []
        self.nodeNum = 1
        self.paths = []
        self.diffs = []
        self.maxDiff = 1
        self.loops = 0
        self.baseWeight = baseWeight
        self.times = [0, 0, 0, 0]  # time spent while in [activation, backPropagate, weightUpdate, overall training]
        # create node map, first makes all nodes
        for h in range(len(nodeMap)):  # for each layer
            self.nodes = []
            for i in range(nodeMap[h]):  # for each node in each layer
                position = (h, i)           # to give each node its position
                nodeIds = [self.nodeNum, position]   # (overallNodeNum, [layerNum, nodeNum on layer])
                self.nodes.append(Node(nodeIds))  # create input node
                self.nodeNum += 1
            self.layers.append(self.nodes)
        # then link all nodes with new paths
        i = 0                               # i is layer index
        for layer in self.layers:           # for each layer
            if i < len(self.layers) - 1:  # if not output node
                nextLayer = self.layers[i + 1]  # save next layer
                for node in layer:          # for each node in current layer
                    for nextNode in nextLayer:  # iterate over each node in the next layer
                        node.outputs.append(nextNode)   # connect the nodes to eachother
                        nextNode.inputs.append(node)
                        if not self.baseWeight:     # if no given weight, initialize weights randomly from 0.001 - 1.5
                            randomWeight = random.randint(1, 1500)/1000
                            self.paths.append([node, nextNode, randomWeight])
                        else:
                            self.paths.append([node, nextNode, self.baseWeight])  # create path list with weights
            i += 1

    def activate(self, xlist):
        startT1 = time.time()
        x = 0       # using x for xlist index and z for path weights index
        for layer in self.layers:
            for node in layer:
                if node.position[0] == 0:       # if input layer
                    node.a = xlist[x]
                else:                           # if regular layer
                    sumInVal = 0
                    for i in node.inputs:  # for each connected input node to this node
                        currentPath = [p for p in self.paths if p[0].nodeId == i.nodeId and p[1].nodeId == node.nodeId]
                        # print(currentPath)
                        sumInVal += i.a * currentPath[0][2]
                    node.a = 1 / (1 + (e ** -(sumInVal + node.bias)))   # sigmoid activation
                x += 1
        endT1 = time.time()
        self.times[0] += endT1 - startT1

    def backPropagate(self, y):
        startT2 = time.time()
        ylist = [y]   # TEMPORARY, make call have ylist later to handle other kinds of datasets
        # back propagate error, then update weights
        # get output layer error first then move in
        self.layers.reverse()  # reversing to go backwards easily, will undo after loop
        self.paths.reverse()
        i, n, x = 0, 0, 0      # using i as layer index, n as node index, x as input index
        for layer in self.layers:
            if i == 0:    # if output layer, calc and set those deltas
                for lastlayerNode in layer:
                    for y in ylist:
                        a = lastlayerNode.a
                        lastlayerNode.delta = (y - a) * (a * (1 - a))
                        # print(f" delta{lastlayerNode.nodeId} = {lastlayerNode.delta}")  # shows output nodes delta
            elif 0 < i < len(self.layers) - 1:  # else if a hidden layer
                n = 0
                for node in layer:      # calc and set hidden layer deltas
                    sumVal = 0
                    for output in node.outputs:
                        weight = self.paths[x][2]
                        sumVal += weight * output.delta
                        x += 1
                    a = node.a
                    node.delta = (a * (1 - a)) * sumVal
                    # print(f" delta{node.nodeId} = {node.delta}")    # shows hidden nodes delta
                    n += 1
            i += 1
        self.layers.reverse()
        self.paths.reverse()
        endT2 = time.time()
        self.times[1] += endT2 - startT2

    def weightUpdate(self, lr):
        startT3 = time.time()
        self.diffs = []             # update path weights first
        for path in self.paths:     # for each node path
            inNode = path[0]
            outNode = path[1]
            weight = path[2]
            newWeight = weight + (lr * inNode.a * outNode.delta)
            wDiff = abs(newWeight - weight)
            self.diffs.append(wDiff)
            path[2] = newWeight
            # print(f"wDiff= {wDiff} from node{path[0].nodeId}--{path[2]}-->node{path[1].nodeId}")
        for layer in self.layers:     # then update node biases
            for node in layer:
                if len(node.inputs) > 0:
                    bias = node.bias
                    newBias = bias + (lr * node.delta)
                    self.diffs.append(abs(newBias - bias))
                    node.bias = newBias
                    # print(abs(newBias - bias))
        endT3 = time.time()
        self.times[2] += endT3 - startT3

    def train(self, dataset, lr=0.15, maxLoops=10**5, convg=10**-5):
        startT4 = time.time()
        self.loops = 0                  # loop until difference between weights is less than convg or loop limit
        while self.loops < maxLoops and self.maxDiff > convg:
            for coords in dataset:      # for each set of data points or coordinates
                xlist = coords[:-1]     # separate x values and y values
                y = coords[-1]
                self.activate(xlist)    # activate, pushing values thru nodes
                self.backPropagate(y)   # back propagate error to fill delta values
                self.weightUpdate(lr)   # then update weights using new deltas(error), also fills diffs list
                self.maxDiff = max(self.diffs)  # use max or average of all diffs for converge check
                # print(self.maxDiff)
            self.loops += 1
        endT4 = time.time()
        self.times[3] += endT4 - startT4

    def test(self, xlist):
        self.activate(xlist)
        result = self.layers[-1][-1].a
        # print(result, end='')
        return result

    def __str__(self):
        retString = "---Model structure---\n"           # print model structure by layers
        for layer in self.layers:
            for node in layer:
                retString += f"node {node.nodeId} at {node.position}\n"
                retString += f"     bias = {node.bias}, delta = {node.delta} \n"
                retString += f"     outVal = {node.a} \n"
        retString += "-------paths-------\n"            # print model structure by node connections
        for node in self.paths:
            retString += f"node {node[0].nodeId} --{node[2]}--> node {node[1].nodeId}\n"
        return retString


# input layer of nodeMap must match num of x values in dataset, and last layer must match number of y values
# only works with single output data for now, so last layer of nodeMap must be 1
# can get 100% accuracy with truth table from: AND, OR, XOR, NAND, NOR, XNOR
gates = {'AND': [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],
        'OR': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        'XOR': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        'NAND': [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
        'NOR': [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
        'XNOR': [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]}

numTotal = 0
for dataset in gates.items():
    # for model, default baseWeight is random value between 0.001 and 1.5
    net = Model(nodeMap=[2, 3, 1])     # nodeMap = [input layer, ..., hidden layer(s), ..., output layer]
    print(f"training with {dataset[0]} gate...")
    net.train(dataset[1])           # for train, default lr is 0.15, default maxloops 10**5, default convg 10**-5
    print(f"training took {net.loops} loops")
    print(f"training times in seconds:"
          f"\n  activation={net.times[0]}, back propagate={net.times[1]}, weight update={net.times[2]}"
          f"\n  overhead={net.times[3] - (net.times[0] + net.times[1] + net.times[2])}, total={net.times[3]}")
    # print(net)
    print(f"now testing {dataset[0]} gate...")
    correct = 0
    for coords in dataset[1]:  # for each set of data points or coordinates
        xlist = coords[:-1]  # separate x values and y values
        y = coords[len(coords) - 1]
        print(f" predicted result of {xlist} is {net.test(xlist).__round__()}, true answer is {y}")
        if net.test(xlist).__round__() == y:
            correct += 1
    print(f'accuracy = {correct/len(dataset[1]) * 100}%')
    if correct/len(dataset[1]) == 1:
        numTotal += 1
    print("--------------------------------------------")
print(f'acheived 100% accuracy with {numTotal}/{len(gates)} of the gates')
