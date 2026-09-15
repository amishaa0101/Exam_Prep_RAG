# Algorithms and Data Structures - Course Notes

## 1.1 Sorting Algorithms Overview

Sorting is the process of arranging elements in a specific order (ascending or descending). Sorting algorithms are fundamental to computer science and appear in almost every software system. The choice of sorting algorithm depends on factors like input size, memory constraints, and whether the data is partially sorted.

Sorting algorithms can be classified as comparison-based (merge sort, quick sort, heap sort) or non-comparison-based (counting sort, radix sort, bucket sort). Comparison-based sorts have a theoretical lower bound of O(n log n).

## 1.2 Merge Sort

Merge sort is a divide-and-conquer algorithm invented by John von Neumann in 1945. It works by:
1. Dividing the unsorted array into n sub-arrays, each containing one element
2. Repeatedly merging sub-arrays to produce sorted sub-arrays
3. Continuing until there is only one sub-array remaining

Time Complexity: O(n log n) in ALL cases (best, average, and worst). This is because the array is always divided in half (log n levels) and merging takes O(n) at each level.

Space Complexity: O(n) additional space is needed for the temporary arrays during merging. This makes merge sort NOT an in-place algorithm.

Merge sort is a STABLE sorting algorithm, meaning elements with equal keys maintain their relative order. This property is important when sorting records by multiple fields.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## 1.3 Quick Sort

Quick sort is another divide-and-conquer algorithm, developed by Tony Hoare in 1959. It works by selecting a pivot element and partitioning the array around it.

The partition step places the pivot in its correct sorted position and ensures all elements smaller than the pivot are to its left, and all larger elements are to its right.

Time Complexity:
- Best case: O(n log n) when the pivot divides the array into roughly equal halves
- Average case: O(n log n) with random pivot selection
- Worst case: O(n^2) when the pivot is always the smallest or largest element (already sorted input with first-element pivot)

Space Complexity: O(log n) for the recursive call stack. Quick sort is an IN-PLACE algorithm (does not need extra arrays for data), which is a significant advantage over merge sort.

Quick sort is NOT stable by default. Equal elements may change their relative order during partitioning.

Pivot selection strategies:
- First element: simple but leads to O(n^2) on sorted input
- Last element: same issue
- Random element: gives expected O(n log n)
- Median-of-three: choose median of first, middle, last elements. Good practical choice.

```python
def quick_sort(arr, low, high):
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

## 1.4 Comparison: Merge Sort vs Quick Sort

| Property | Merge Sort | Quick Sort |
|----------|-----------|------------|
| Best Case | O(n log n) | O(n log n) |
| Average Case | O(n log n) | O(n log n) |
| Worst Case | O(n log n) | O(n^2) |
| Space | O(n) | O(log n) |
| Stable | Yes | No |
| In-place | No | Yes |
| Cache friendly | No | Yes |

In practice, quick sort is often faster than merge sort despite having a worse worst case, because it has better cache locality (works on contiguous memory) and smaller constant factors. However, merge sort is preferred when stability is required or when working with linked lists (where the space disadvantage disappears).

## 2.1 Graph Algorithms Overview

A graph G = (V, E) consists of vertices V and edges E. Graphs can be directed or undirected, weighted or unweighted. Graph algorithms are used in networking, social media, maps, and many other applications.

Key representations:
- Adjacency Matrix: O(V^2) space, O(1) edge lookup
- Adjacency List: O(V + E) space, O(degree) edge lookup

## 2.2 Breadth-First Search (BFS)

BFS explores a graph level by level, visiting all neighbors of a vertex before moving to the next level. It uses a QUEUE data structure.

Algorithm:
1. Start at source vertex, mark as visited, enqueue it
2. While queue is not empty:
   a. Dequeue a vertex v
   b. For each unvisited neighbor u of v:
      - Mark u as visited
      - Record the distance/parent
      - Enqueue u

Time Complexity: O(V + E) where V is vertices and E is edges.
Space Complexity: O(V) for the visited array and queue.

BFS finds SHORTEST PATHS in unweighted graphs. The distance from source to any vertex is the number of edges in the BFS tree path.

Applications: shortest path in unweighted graphs, social network friend suggestions (people at distance 2), web crawling, checking bipartiteness.

```python
from collections import deque

def bfs(graph, source):
    visited = {source}
    queue = deque([source])
    distances = {source: 0}

    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)

    return distances
```

## 2.3 Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path from a single source to all other vertices in a WEIGHTED graph with NON-NEGATIVE edge weights.

It uses a PRIORITY QUEUE (min-heap) to always process the vertex with the smallest known distance.

Algorithm:
1. Initialize distances: source = 0, all others = infinity
2. Add source to priority queue
3. While priority queue is not empty:
   a. Extract vertex u with minimum distance
   b. For each neighbor v of u:
      - If dist[u] + weight(u,v) < dist[v]:
        - Update dist[v] = dist[u] + weight(u,v)
        - Add v to priority queue

Time Complexity: O((V + E) log V) with a binary heap implementation.
Space Complexity: O(V) for distances and the priority queue.

IMPORTANT: Dijkstra's algorithm does NOT work with negative edge weights. This is because once a vertex is "finalized" (extracted from the priority queue), Dijkstra assumes its distance is optimal. A negative edge later could invalidate this assumption. For graphs with negative weights, use Bellman-Ford algorithm.

```python
import heapq

def dijkstra(graph, source):
    distances = {v: float('inf') for v in graph}
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        dist_u, u = heapq.heappop(pq)

        if dist_u > distances[u]:
            continue  # skip outdated entry

        for v, weight in graph[u]:
            new_dist = dist_u + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return distances
```

## 2.4 Bellman-Ford Algorithm

Bellman-Ford algorithm also solves single-source shortest path, but unlike Dijkstra, it handles NEGATIVE edge weights and can detect NEGATIVE CYCLES.

Algorithm:
1. Initialize distances: source = 0, all others = infinity
2. Repeat V-1 times:
   - For each edge (u, v) with weight w:
     - If dist[u] + w < dist[v]: update dist[v]
3. Check for negative cycles: repeat step 2 once more. If any distance decreases, a negative cycle exists.

Time Complexity: O(V * E) which is slower than Dijkstra for non-negative graphs.
Space Complexity: O(V).

Use Bellman-Ford when: graph has negative weights, you need to detect negative cycles, or the graph is small enough that the O(VE) cost is acceptable.

## 3.1 Dynamic Programming

Dynamic programming (DP) solves complex problems by breaking them into simpler overlapping subproblems. It stores the results of subproblems to avoid redundant computation (memoization or tabulation).

Two key properties that make a problem suitable for DP:
1. Optimal Substructure: optimal solution can be constructed from optimal solutions of subproblems
2. Overlapping Subproblems: the same subproblems are solved multiple times

Two approaches:
- Top-down (Memoization): recursive with caching. More intuitive, follows the recurrence directly.
- Bottom-up (Tabulation): iterative, fills a table from base cases up. Usually more space-efficient and avoids recursion overhead.

## 3.2 Fibonacci as a DP Example

The Fibonacci sequence: F(0)=0, F(1)=1, F(n) = F(n-1) + F(n-2)

Naive recursive: O(2^n) time, O(n) space. Exponential because it recomputes the same values.
DP with memoization: O(n) time, O(n) space. Each subproblem computed once.
DP with tabulation: O(n) time, O(1) space (only need last two values).

```python
# Naive recursive - O(2^n) - DON'T use this
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)

# Top-down with memoization - O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Bottom-up tabulation - O(n) time, O(1) space
def fib_tab(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n+1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```
