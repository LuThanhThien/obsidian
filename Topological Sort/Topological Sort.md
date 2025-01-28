Many real world applications can be modeled as a graph with directed edges where some event must occur before others.
- School class prerequisites
- Program dependencies
- Event scheduling
- Assembly instruction 
![[Topological Sort Graph | 700 | center]]
# **Definition**
**Topological Sorted order**: It is a linear ordering of vertices such that for every directed edge ***u -> v***, where vertex ***u*** comes before ***v*** in the ordering.
- A **topological sort algorithm** can find a topological ordering in `O(V+E)` time!
- Topological orderings are **not unique**.
1. **Directed Acyclic Graph:** Not all graphs have topological sorted order, for example graph that has relationships between nodes in cycles. The only type of graph which has a valid topological ordering is a **[[Directed Acyclic Graph]]** - **graph with directed edges with no cycles**.
![[Circular dependent graph | 400 | center]]
	**Q:** How to identify whether a graph is Directed Acyclic Graph or not?
	**A:** One method is to use **[[Tarjan's Strongly Connected Component (SCC)]]** to find those cycles in a graph.
2. All rooted trees have topological ordering since they do not contain any cycles. We can start from the children and work up to the root.
![[Rooted Trees | 500 | center]]
# **Topological Sort algorithm**

## **DFS**
We can use **[[Depth First Search (DFS)]]** algorithm for topological sort, in details: 
- Pick an unvisited node.
- Beginning with the selected node, do DFS exploring only unvisited nodes.
- On the recursive callback of the DFS, add the current node to the topological ordering in reverse order.
![[TopSort Flow | center | 250]]
```pseudo
\begin{algorithm}
\caption{Topological Sort (TopSort)}
\begin{algorithmic}
  \PROCEDURE{TopSort}{$G$}
    \STATE $N \gets$ number of nodes in $G$
    \STATE $V \gets$ array of $N$ elements initialized to $false$ \COMMENT{Visited array}
    \STATE $ordering \gets$ array of $N$ elements initialized to $0$
    \STATE $i \gets N - 1$ \COMMENT{Index for the ordering array}
    \FOR{$at = 0$ \TO $N - 1$}
      \IF{$V[at] = false$}
        \STATE \CALL{DFS}{$G, at, V, ordering, i$}
      \ENDIF
    \ENDFOR
    \RETURN $ordering$
  \ENDPROCEDURE
  
  \PROCEDURE{DFS}{$G, node, V, ordering, i$}
    \STATE $V[node] \gets true$ \COMMENT{Marked as visited}
    \FORALL{neighbors $n$ of $node$ in $G$}
      \IF{$V[n] = false$}
        \STATE \CALL{DFS}{$G, n, V, ordering, i$}
      \ENDIF
    \ENDFOR
    \STATE $ordering[i] \gets node$
    \STATE $i \gets i - 1$
  \ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
![[TopSort Example| 1000 | center]]
Time complexity: `O(V + E)` 
Space complexity: `O(V)`
(`V`: number of vertices, `E`: number of edges)
## **Kahn's Algorithm**
**Kahn's Algorithm** for **Topological Sorting** is a method used to order the vertices of a directed graph in a linear order such that for every directed edge from vertex ***A*** to vertex **B**, ***A*** comes before **B** in the order. The algorithm works by **repeatedly finding vertices with no incoming edges**, removing them from the graph, and updating the incoming edges of the remaining vertices. This process continues until all vertices have been ordered.

```pseudo
\begin{algorithm}
\caption{Kahn's Algorithm for Topological Sort}
\begin{algorithmic}
  \PROCEDURE{FindTopologicalOrdering}{$G$}
    \STATE $n \gets$ number of nodes in $G$
    \STATE $in\_degree \gets$ array of size $n$, initialized to $0$ \COMMENT{Stores in-degrees of nodes}
    \FOR{$i = 0$ \TO $n - 1$}
      \FORALL{$to$ in $G[i]$}
        \STATE $in\_degree[to] \gets in\_degree[to] + 1$
      \ENDFOR
    \ENDFOR

    \STATE $q \gets$ empty queue \COMMENT{Queue to store nodes with in-degree 0}
    \FOR{$i = 0$ \TO $n - 1$}
      \IF{$in\_degree[i] = 0$}
        \STATE $q.\text{enqueue}(i)$
      \ENDIF
    \ENDFOR

    \STATE $index \gets 0$
    \STATE $order \gets$ array of size $n$, initialized to $0$
    
    \WHILE{$!q.\text{isEmpty}()$}
      \STATE $at \gets q.\text{dequeue}()$
      \STATE $order[index] \gets at$
      \STATE $index \gets index + 1$
      \FORALL{$to$ in $G[at]$}
        \STATE $in\_degree[to] \gets in\_degree[to] - 1$
        \IF{$in\_degree[to] = 0$}
          \STATE $q.\text{enqueue}(to)$
        \ENDIF
      \ENDFOR
    \ENDWHILE

    \IF{$index \neq n$}
      \RETURN null \COMMENT{Graph contains a cycle}
    \ENDIF

    \RETURN $order$
  \ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
# **References**
https://www.youtube.com/watch?v=eL-KzMXSXXI
https://www.youtube.com/watch?v=cIBFEhD77b4