import numpy as np
from Optimization.Algorithm import sampling as sp

"""""""""
This file contains the code for the neldermead algorithm. This is a simplex method which requires no derivative 
and converges on local minimizers for sufficiently smooth functions.

This algorithm takes the function class input from the classy.py file.

You may also include constraints through the funct.para.boundary class location to include a function which returns True
if the function is feasible and returns False if not.
"""""""""


def scatter(input, points):
    # Create initial simplex from initial input
    input = np.array(input)
    scat = []
    sampling = sp.halton(points, len(input), 12345678)
    for k in range(points):
        scat.append(input + sampling[k].flatten())
    return scat


def sort(array):
    # Determine best and worst points
    for j in range(len(array)):
        if array[j][1] < array[0][1]:
            array[0], array[j] = array[j], array[0]
    for j in range(len(array)):
        if array[j][1] > array[len(array) - 1][1]:
            array[len(array) - 1], array[j] = array[j], array[len(array) - 1]
    return array


def centroid(array):
    # Find the centroid
    center = 0
    for j in range(len(array)):
        center += np.array(array[j][0]) / len(array)
    return center


def conditions(array, centroid):
    # Check for stopping criteria, average distance from centroid
    distance = 0
    for j in range(len(array)):
        distance += np.linalg.norm(centroid - array[j][0]) / len(array)
    if distance < 10 ** (-9.5):
        return True
    return False


def boundary(input, para):
    # Check for constraint violation
    if para.boundary(input):
        return True
    return False



def neldermead(pr):
    # Initialize sample
    nodes = []
    size = len(pr.input) + 1
    sample = scatter(pr.input, size)
    # Store input and outputs together in 'nodes'
    for i in range(size):
        if pr.para.boundary(sample[i]):
            nodes.append([sample[i], pr.function(sample[i], pr.para, 0)])
        else:
            return print('Initial Values not in Domain')
    l = len(nodes) - 1
    # Iterate
    for k in range(1000000):
        # Sort nodes
        nodes = sort(nodes)
        # Find second best value
        second = max(nodes[0:l][j][1] for j in range(l))
        # Find centroid
        center = centroid(nodes)
        # Print progress
        if k % pr.print == 0:
            print(str(k) + '___' + str(nodes[0][0]) + '___' + str(nodes[0][1]))
        # Check for stopping criteria
        if conditions(nodes, center):
            print('Simplex has achieved minimal size')
            print(str(k) + '___' + str(nodes[0][0]) + '___' + str(nodes[0][1]))
            return
        # Search for improvement to worst point
        reflect = 2 * center - nodes[l][0]
        eval = pr.function(reflect, pr.para, 0)
        if eval < nodes[0][1]:
            expand = 3 * center - 2 * np.array(nodes[l][0])
            expeval = pr.function(expand, pr.para, 0)
            if expeval < eval:
                if pr.para.boundary(sample[i]):
                    nodes[l][0], nodes[l][1] = expand, expeval
            else:
                if pr.para.boundary(sample[i]):
                    nodes[l][0], nodes[l][1] = reflect, eval
        if nodes[0][1] <= eval < second:
            if pr.para.boundary(sample[i]):
                nodes[l][0], nodes[l][1] = reflect, eval
        if second <= eval < nodes[l][1]:
            outside = 1 / 2 * (center + reflect)
            outeval = pr.function(outside, pr.para, 0)
            if outeval < eval:
                if pr.para.boundary(sample[i]):
                    nodes[l][0], nodes[l][1] = outside, outeval
            else:
                if pr.para.boundary(sample[i]):
                    nodes[l][0], nodes[l][1] = reflect, eval
        if nodes[l][1] <= eval:
            inside = 1 / 2 * (center + nodes[l][0])
            ineval = pr.function(inside, pr.para, 0)
            if ineval < nodes[l][1]:
                if pr.para.boundary(sample[i]):
                    nodes[l][0], nodes[l][1] = inside, ineval
            else:
                for j in range(l):
                    newnode = nodes[j + 1][0] + 1 / 2 * (nodes[j + 1][0] - nodes[0][0])
                    nodes[j + 1] = [newnode, pr.function(newnode, pr.para, 0)]
    return print('Failed to converge')
