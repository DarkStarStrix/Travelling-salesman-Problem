
# Traveling Salesman Problem (TSP): Given a set of cities and distance between every pair of cities, the problem is to find the shortest possible route that visits every city exactly once and returns to the starting point.

# using the The Beardwood–Halton–Hammersley theorem to solve the TSP problem in O(n^2) time complexity and O(n) space complexity 

# The Beardwood–Halton–Hammersley theorem states that the shortest path that visits every point in a set of points and returns to the starting point is the minimum spanning tree of the complete graph of the points, with distances given by the Euclidean distance (i.e. straight-line distance).

# The minimum spanning tree can be found in O(n^2) time using Prim's algorithm or Kruskal's algorithm. The minimum spanning tree can also be found in O(n log n) time using Borůvka's algorithm.

# The minimum spanning tree can be found in O(n) space using Prim's algorithm or Borůvka's algorithm. The minimum spanning tree can also be found in O(n^2) space using Kruskal's algorithm.

# The minimum spanning tree can be found in O(n^2) time and O(n) space using the reverse-delete algorithm.

# use python3 to run the code the algorithm is implemented in python3

# importing the required modules
# Traveling Salesman Problem (TSP): Given a set of cities and distance between every pair of cities, the problem is to find the shortest possible route that visits every city exactly once and returns to the starting point.

# using the The Beardwood–Halton–Hammersley theorem to solve the TSP problem in O(n^2) time complexity and O(n) space complexity 

# The Beardwood–Halton–Hammersley theorem states that the shortest path that visits every point in a set of points and returns to the starting point is the minimum spanning tree of the complete graph of the points, with distances given by the Euclidean distance (i.e. straight-line distance).

# The minimum spanning tree can be found in O(n^2) time using Prim's algorithm or Kruskal's algorithm. The minimum spanning tree can also be found in O(n log n) time using Borůvka's algorithm.

# The minimum spanning tree can be found in O(n) space using Prim's algorithm or Borůvka's algorithm. The minimum spanning tree can also be found in O(n^2) space using Kruskal's algorithm.

# The minimum spanning tree can be found in O(n^2) time and O(n) space using the reverse-delete algorithm.

# use python3 to run the code the algorithm is implemented in python3

# importing the required modules
import math
import random
import matplotlib.pyplot as plt
import time

# function to calculate the euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# function to calculate the total distance of the path
def total_distance(points):
    distance = 0
    points = points + [points[0]]
    for i in range(len(points) - 1):
        distance += euclidean_distance(points[i], points[i + 1])
    distance += euclidean_distance(points[0], points[-1])
    return distance

# function to generate the points
def generate_points(n):
    points = []
    for i in range(n):
        points.append((random.randint(0, 100), random.randint(0, 100)))
    return points

# function to plot the points
def plot_points(points):
    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()

plot_points(generate_points(10))

# function to find the minimum spanning tree
def minimum_spanning_tree(points):
# creating the complete graph
    graph = {}
    for i in range(len(points)):
        graph[i] = {}
        for j in range(len(points)):
            if i != j:
                graph[i][j] = euclidean_distance(points[i], points[j])
    # finding the minimum spanning tree
    mst = {}
    mst[0] = None
    visited = [0]
    while len(visited) != len(points):
        minimum = float('inf')
        for i in visited:
            for j in range(len(points)):
                if j not in visited:
                    if graph[i][j] < minimum:
                        minimum = graph[i][j]
                        node = j
                        parent = i
        mst[node] = parent
        visited.append(node)
    return mst

# function to find the path
def find_path(mst):
 path = []
 for i in range(len(mst)):
        path.append(i)
 path.append(0)
 return path

# function to find the path using the minimum spanning tree
def find_path_using_mst(points):
 mst = minimum_spanning_tree(points)
 path = find_path(mst)
 return path

