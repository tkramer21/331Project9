import math
import os
import string
import unittest

from solution import Graph, Vertex


class GraphTests(unittest.TestCase):
    """
    Begin Vertex Tests
    """

    def test_deg(self):
        # (1) test a-->b and a-->c
        vertex = Vertex('a')
        vertex.adj['b'] = 1
        self.assertEqual(1, vertex.deg())
        vertex.adj['c'] = 3
        self.assertEqual(2, vertex.deg())

        # (2) test a-->letter for all letters in alphabet
        vertex = Vertex('a')
        for i, char in enumerate(string.ascii_lowercase):
            self.assertEqual(vertex.deg(), i)
            vertex.adj[char] = i

    def test_get_outgoing_edges(self):

        # (1) test a-->b and a-->c
        vertex = Vertex('a')
        solution = {('b', 1), ('c', 2)}
        vertex.adj['b'] = 1
        vertex.adj['c'] = 2
        subject = vertex.get_outgoing_edges()
        self.assertEqual(subject, solution)

        # (2) test empty case
        vertex = Vertex('a')
        solution = set()
        subject = vertex.get_outgoing_edges()
        self.assertEqual(subject, solution)

        # (3) test a-->letter for all letters in alphabet
        for i, char in enumerate(string.ascii_lowercase):
            vertex.adj[char] = i
            solution.add((char, i))
        subject = vertex.get_outgoing_edges()
        self.assertEqual(subject, solution)

    def test_distances(self):

        # (1) test pythagorean triple
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 3, 4)

        subject = vertex_a.euclidean_dist(vertex_b)
        self.assertEqual(subject, 5)
        subject = vertex_b.euclidean_dist(vertex_a)
        self.assertEqual(subject, 5)
        subject = vertex_a.taxicab_dist(vertex_b)
        self.assertEqual(subject, 7)
        subject = vertex_b.taxicab_dist(vertex_a)
        self.assertEqual(subject, 7)

        # (2) test linear difference
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 0, 10)

        subject = vertex_a.euclidean_dist(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.euclidean_dist(vertex_a)
        self.assertEqual(subject, 10)
        subject = vertex_a.taxicab_dist(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.taxicab_dist(vertex_a)
        self.assertEqual(subject, 10)

        # (3) test zero distance
        vertex_a = Vertex('a')
        vertex_b = Vertex('b')

        subject = vertex_a.euclidean_dist(vertex_b)
        self.assertEqual(subject, 0)
        subject = vertex_b.euclidean_dist(vertex_a)
        self.assertEqual(subject, 0)
        subject = vertex_a.taxicab_dist(vertex_b)
        self.assertEqual(subject, 0)
        subject = vertex_b.taxicab_dist(vertex_a)
        self.assertEqual(subject, 0)

        # (4) test floating point distance
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 5, 5)

        subject = vertex_a.euclidean_dist(vertex_b)
        self.assertAlmostEqual(subject, 5 * math.sqrt(2))
        subject = vertex_b.euclidean_dist(vertex_a)
        self.assertAlmostEqual(subject, 5 * math.sqrt(2))
        subject = vertex_a.taxicab_dist(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.taxicab_dist(vertex_a)
        self.assertEqual(subject, 10)

        # (5) test taxicab absolute values in right spot
        vertex_a = Vertex('a', 3, 1)
        vertex_b = Vertex('b', 2, 5)
        subject = vertex_a.euclidean_dist(vertex_b)
        self.assertAlmostEqual(subject, math.sqrt(17))
        subject = vertex_b.euclidean_dist(vertex_a)
        self.assertAlmostEqual(subject, math.sqrt(17))
        subject = vertex_a.taxicab_dist(vertex_b)
        self.assertEqual(subject, 5)
        subject = vertex_b.taxicab_dist(vertex_a)
        self.assertEqual(subject, 5)

    """
    End Vertex Tests
    """

    """
    Begin Graph Tests
    """

    def test_unvisit_vertices(self):

        graph = Graph()

        # (1) visit all vertices then reset
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')

        for vertex in graph.vertices.values():
            vertex.visited = True
        graph.unvisit_vertices()
        for vertex in graph.vertices.values():
            self.assertFalse(vertex.visited)

    def test_get_vertex_by_id(self):

        graph = Graph()

        # (1) test basic vertex object
        vertex_a = Vertex('a')
        graph.vertices['a'] = vertex_a
        subject = graph.get_vertex_by_id('a')
        self.assertEqual(subject, vertex_a)

        # (2) test empty case
        subject = graph.get_vertex_by_id('b')
        self.assertIsNone(subject)

        # (3) test case with adjacencies
        vertex_b = Vertex('b')
        for i, char in enumerate(string.ascii_lowercase):
            vertex_b.adj[char] = i
        graph.vertices['b'] = vertex_b
        subject = graph.get_vertex_by_id('b')
        self.assertEqual(subject, vertex_b)

    def test_get_all_vertices(self):

        graph = Graph()
        solution = set()

        # (1) test empty graph
        subject = graph.get_all_vertices()
        self.assertEqual(subject, solution)

        # (2) test single vertex
        vertex = Vertex('$')
        graph.vertices['$'] = vertex
        solution.add(vertex)
        subject = graph.get_all_vertices()
        self.assertEqual(subject, solution)

        # (3) test multiple vertices
        graph = Graph()
        solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            vertex = Vertex(char)
            graph.vertices[char] = vertex
            solution.add(vertex)
        subject = graph.get_all_vertices()
        self.assertEqual(subject, solution)

    def test_get_edge_by_ids(self):

        graph = Graph()

        # (1) neither end vertex exists
        subject = graph.get_edge_by_ids('a', 'b')
        self.assertIsNone(subject)

        # (2) one end vertex exists
        graph.vertices['a'] = Vertex('a')
        subject = graph.get_edge_by_ids('a', 'b')
        self.assertIsNone(subject)

        # (3) both end vertices exist, but no edge
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        subject = graph.get_edge_by_ids('a', 'b')
        self.assertIsNone(subject)

        # (4) a -> b exists but b -> a does not
        graph.vertices.get('a').adj['b'] = 331
        subject = graph.get_edge_by_ids('a', 'b')
        self.assertEqual(subject, ('a', 'b', 331))
        subject = graph.get_edge_by_ids('b', 'a')
        self.assertIsNone(subject)

        # (5) connect all vertices to center vertex and return all edges
        graph.vertices['$'] = Vertex('$')
        for i, char in enumerate(string.ascii_lowercase):
            graph.vertices[char] = Vertex(char)
            graph.vertices.get('$').adj[char] = i
        for i, char in enumerate(string.ascii_lowercase):
            subject = graph.get_edge_by_ids('$', char)
            self.assertEqual(subject, ('$', char, i))

    def test_get_all_edges(self):

        graph = Graph()

        # (1) test empty graph
        subject = graph.get_all_edges()
        self.assertEqual(subject, set())

        # (2) test graph with vertices but no edges
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        subject = graph.get_all_edges()
        self.assertEqual(subject, set())

        # (3) test graph with one edge
        graph.vertices.get('a').adj['b'] = 331
        subject = graph.get_all_edges()
        self.assertEqual(subject, {('a', 'b', 331)})

        # (4) test graph with two edges
        graph = Graph()
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        graph.vertices.get('a').adj['b'] = 331
        graph.vertices.get('b').adj['a'] = 1855
        subject = graph.get_all_edges()
        solution = {('a', 'b', 331), ('b', 'a', 1855)}
        self.assertEqual(subject, solution)

        # (5) test entire alphabet graph
        graph = Graph()
        solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            graph.vertices[char] = Vertex(char)
            for j, jar in enumerate(string.ascii_lowercase):
                if i != j:
                    graph.vertices.get(char).adj[jar] = 26 * i + j
                    solution.add((char, jar, 26 * i + j))
        #graph.size = len(graph.vertices)
        #graph.plot_show = True
        #graph.plot()
        subject = graph.get_all_edges()
        self.assertEqual(subject, solution)

    def test_build_path(self):

        # (1) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph._build_path({'b': 'a'}, 'a', 'b')
        self.assertEqual(subject, (['a', 'b'], 331))

        # (2) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph._build_path({'b': 'a', 'c': 'b'}, 'a', 'c')
        self.assertEqual(subject, (['a', 'b', 'c'], 431))

        # (3) test that the right weights are being selected
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        graph.add_to_graph('a', 'c', 999)
        subject = graph._build_path({'c': 'a'}, 'a', 'c')
        self.assertEqual(subject, (['a', 'c'], 999))

        # (4) test on a lot of edges
        graph = Graph(csvf='graphs_csv/bfs/7.csv')
        subject = graph._build_path({'midleft': 'bottomleft', 'topleft': 'midleft', 'topright': 'topleft'},
                                    'bottomleft',
                                    'topright')
        self.assertEqual(subject, (['bottomleft', 'midleft', 'topleft', 'topright'], 3))

    def test_bfs(self):
        graph = Graph()

        # (1) test on empty graph
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (2) test on graph missing begin or dest
        graph.add_to_graph('a')
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))
        subject = graph.bfs('b', 'a')
        self.assertEqual(subject, ([], 0))

        # (3) test on graph with both vertices but no path
        graph.add_to_graph('b')
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (4) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, (['a', 'b'], 331))

        # (5) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph.bfs('a', 'c')
        self.assertEqual(subject, (['a', 'b', 'c'], 431))

        # (6) test on edge triangle and ensure one-edge path is taken
        # (bfs guarantees fewest-edge path, not least-weighted path)
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        graph.add_to_graph('a', 'c', 999)
        subject = graph.bfs('a', 'c')
        self.assertEqual(subject, (['a', 'c'], 999))

        # (7) test on grid figure-8 and ensure fewest-edge path is taken
        graph = Graph(csvf='graphs_csv/bfs/7.csv')

        subject = graph.bfs('bottomleft', 'topleft')
        self.assertEqual(subject, (['bottomleft', 'midleft', 'topleft'], 2))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('bottomright', 'topright')
        self.assertEqual(subject, (['bottomright', 'midright', 'topright'], 2))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('bottomleft', 'topright')
        self.assertIn(subject[0], [['bottomleft', 'midleft', 'topleft', 'topright'],
                                   ['bottomleft', 'midleft', 'midright', 'topright'],
                                   ['bottomleft', 'bottomright', 'midright', 'topright']])
        self.assertEqual(subject[1], 3)

        # (8) test on example graph from Onsay's slides, starting from vertex A
        # see bfs_graph.png
        graph = Graph(csvf='graphs_csv/bfs/8.csv')

        subject = graph.bfs('a', 'd')
        self.assertEqual(subject, (['a', 'b', 'd'], 4))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'f')
        self.assertEqual(subject, (['a', 'c', 'f'], 4))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'h')
        self.assertEqual(subject, (['a', 'e', 'h'], 4))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'g')
        self.assertEqual(subject, (['a', 'e', 'g'], 4))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'i')
        self.assertIn(subject[0], [['a', 'e', 'h', 'i'], ['a', 'e', 'g', 'i']])
        self.assertEqual(subject[1], 6)

        # (9) test path which does not exist
        graph.unvisit_vertices()  # mark all unvisited
        graph.add_to_graph('z')
        subject = graph.bfs('a', 'z')
        self.assertEqual(subject, ([], 0))

    def test_dfs(self):

        graph = Graph()

        # (1) test on empty graph
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (2) test on graph missing begin or dest
        graph.add_to_graph('a')
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))
        subject = graph.dfs('b', 'a')
        self.assertEqual(subject, ([], 0))

        # (3) test on graph with both vertices but no path
        graph.add_to_graph('b')
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (4) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, (['a', 'b'], 331))

        # (5) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph.dfs('a', 'c')
        self.assertEqual(subject, (['a', 'b', 'c'], 431))

        # (6) test on linear chain with backtracking distract
        # see linear_graph.png
        graph = Graph(csvf='graphs_csv/dfs/6.csv')

        subject = graph.dfs('a', 'e')
        self.assertEqual(subject, (['a', 'b', 'c', 'd', 'e'], 4))

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.dfs('e', 'a')
        self.assertEqual(subject, (['e', 'd', 'c', 'b', 'a'], 8))

        # (7) test on linear chain with cycle
        # see cyclic_graph.png
        graph = Graph(csvf='graphs_csv/dfs/7.csv')

        subject = graph.dfs('a', 'd')
        graph.plot_show = True
        graph.plot()
        self.assertIn(subject, [(['a', 'b', 'm', 'c', 'd'], 24),
                                (['a', 'b', 'n', 'c', 'd'], 28)])

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.dfs('d', 'a')
        self.assertIn(subject, [(['d', 'c', 'm', 'b', 'a'], 240),
                                (['d', 'c', 'n', 'b', 'a'], 280)])

        # (8) test path which does not exist on graph
        graph.unvisit_vertices()
        graph.add_to_graph('z')
        subject = graph.dfs('a', 'z')
        self.assertEqual(subject, ([], 0))

    def test_topological_sort(self):

        graph = Graph()

        # (1) test on empty graph
        subject = graph.topological_sort()
        self.assertEqual(subject, [])

        # (2) test on lone vertex graph
        graph.add_to_graph('a')
        subject = graph.topological_sort()
        self.assertEqual(subject, ['a'])

        # (3) test on single edge graph
        graph = Graph()
        graph.add_to_graph('a', 'b')
        subject = graph.topological_sort()
        self.assertEqual(subject, ['a', 'b'])

        # (4) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b')
        graph.add_to_graph('b', 'c')
        subject = graph.topological_sort()
        self.assertEqual(subject, ['a', 'b', 'c'])

        # (5) test on transitive graph with three edges
        graph = Graph()
        graph.add_to_graph('a', 'b')
        graph.add_to_graph('b', 'c')
        graph.add_to_graph('a', 'c')
        subject = graph.topological_sort()
        self.assertEqual(subject, ['a', 'b', 'c'])

        # define helper function to check validity of topological_sort result
        def check_if_topologically_sorted(g, topo):
            edges = g.get_all_edges()
            return all(topo.index(edge[0]) < topo.index(edge[1]) for edge in edges)

        # (6) test on two edges -- multiple correct answers from here on out
        graph = Graph()
        graph.add_to_graph('a', 'b')
        graph.add_to_graph('a', 'c')
        subject = graph.topological_sort()
        self.assertTrue(check_if_topologically_sorted(graph, subject))

        # (7) test on small DAG
        graph = Graph()
        graph.add_to_graph('A', 'B')
        graph.add_to_graph('A', 'F')
        graph.add_to_graph('B', 'C')
        graph.add_to_graph('D', 'A')
        graph.add_to_graph('D', 'C')
        graph.add_to_graph('E', 'C')
        graph.add_to_graph('F', 'E')
        subject = graph.topological_sort()
        self.assertTrue(check_if_topologically_sorted(graph, subject))

        # (8) test on medium DAG
        graph = Graph(csvf='graphs_csv/topo/1.csv')
        subject = graph.topological_sort()
        self.assertTrue(check_if_topologically_sorted(graph, subject))

        # (9) test on large DAG
        graph = Graph(csvf='graphs_csv/topo/2.csv')
        subject = graph.topological_sort()
        self.assertTrue(check_if_topologically_sorted(graph, subject))

        # (10) test on very large DAG with many edges
        graph = Graph(csvf='graphs_csv/topo/3.csv')
        subject = graph.topological_sort()
        self.assertTrue(check_if_topologically_sorted(graph, subject))

        # DAGs generated by adapting https://github.com/Willtl/online-printing-shop/blob/master/generator/instance_generator.py

    """
    End Graph Tests
    """

    def test_friend_recommender(self):
        def add_undirected_edge(graph, start, end):
            graph.add_to_graph(start, end)
            graph.add_to_graph(end, start)

        graph = Graph()
        expected = []
        self.assertEqual(expected, graph.friends_recommender(""))  # 1. Test Empty Graph (No vertex)

        graph = Graph()
        add_undirected_edge(graph, 'Shrek', 'Donkey')
        add_undirected_edge(graph, 'Shrek', 'Fiona')
        expected = []
        self.assertEqual(expected, graph.friends_recommender('Shrek'))  # 2. Test all friends

        graph = Graph()
        add_undirected_edge(graph, 'Finn The Human', 'Princess Bubblegum')
        add_undirected_edge(graph, 'Princess Bubblegum', 'Marceline')
        expected = ['Marceline']
        self.assertEqual(expected, graph.friends_recommender('Finn The Human'))  # 3. Test one recommend friend

        graph = Graph()
        add_undirected_edge(graph, 'Sanic', 'Barry B. Benson')
        add_undirected_edge(graph, 'Barry B. Benson', 'Gigachad')
        add_undirected_edge(graph, 'Barry B. Benson', 'Globglogabgalab')
        expected = ['Gigachad', 'Globglogabgalab']
        self.assertEqual(expected,
                         graph.friends_recommender('Sanic'))  # 3. Test two recommend friends with same priority

        graph = Graph()
        add_undirected_edge(graph, 'Shrek', 'Donkey')
        add_undirected_edge(graph, 'Shrek', 'Mike Wazowski')
        add_undirected_edge(graph, 'Donkey', 'Senator Armstrong')
        add_undirected_edge(graph, 'Donkey', 'Doge')
        add_undirected_edge(graph, 'Mike Wazowski', 'Senator Armstrong')
        add_undirected_edge(graph, 'Mike Wazowski', 'Nyan Cat')
        expected = ['Senator Armstrong', 'Doge', 'Nyan Cat']
        self.assertEqual(expected,
                         graph.friends_recommender('Shrek'))  # 3. Test recommend friends with different priorities

        graph = Graph()
        add_undirected_edge(graph, 'Brimstone', 'Neon')
        add_undirected_edge(graph, 'Brimstone', 'KAY/O')
        add_undirected_edge(graph, 'Brimstone', 'Fade')
        add_undirected_edge(graph, 'Neon', 'Skye')
        add_undirected_edge(graph, 'Neon', 'Viper')
        add_undirected_edge(graph, 'KAY/O', 'Viper')
        add_undirected_edge(graph, 'KAY/O', 'Fade')
        add_undirected_edge(graph, 'Fade', 'Skye')
        expected = ['Skye', 'Viper']
        self.assertEqual(expected, graph.friends_recommender('Brimstone'))  # 4. Test more depth

        graph = Graph()
        add_undirected_edge(graph, 'Shrek', 'Donkey')
        add_undirected_edge(graph, 'Shrek', 'Fiona')
        add_undirected_edge(graph, 'Donkey', 'Dragon')
        add_undirected_edge(graph, 'Donkey', 'Puss in Boots')
        add_undirected_edge(graph, 'Fiona', 'Puss in Boots')
        add_undirected_edge(graph, 'Fiona', 'The Big Bad Wolf')
        add_undirected_edge(graph, 'The Big Bad Wolf', 'Three Little Pigs')
        expected = ['Puss in Boots', 'Dragon', 'The Big Bad Wolf', 'Three Little Pigs']
        self.assertEqual(expected, graph.friends_recommender('Shrek'))  # 5. test general

        graph = Graph()
        add_undirected_edge(graph, 'Mr. Krabs', 'Pearl')
        add_undirected_edge(graph, 'Mr. Krabs', 'Squidward')
        add_undirected_edge(graph, 'Spongebob', 'Patrick')
        add_undirected_edge(graph, 'Spongebob', 'Gary')
        add_undirected_edge(graph, 'Spongebob', 'Squilliam')
        expected = []
        self.assertEqual(expected, graph.friends_recommender('Spongebob'))  # 6. Disjoint graph

        graph = Graph()
        add_undirected_edge(graph, 'Red Crewmate', 'Blue Crewmate')
        add_undirected_edge(graph, 'Red Crewmate', 'Green Crewmate')
        add_undirected_edge(graph, 'Magenta Crewmate', 'White Crewmate')
        add_undirected_edge(graph, 'Cyan Crewmate', 'White Crewmate')
        add_undirected_edge(graph, 'Pink Crewmate', 'Cyan Crewmate')
        add_undirected_edge(graph, 'Pink Crewmate', 'Magenta Crewmate')
        add_undirected_edge(graph, 'Red Imposter', 'Pink Crewmate')
        expected = ['Cyan Crewmate', 'Magenta Crewmate', 'White Crewmate']
        self.assertEqual(expected, graph.friends_recommender('Red Imposter'))  # 7. Larger disjoint test
        # Checking reset vertices visit before traversal (GOOD PRACTICE !!!)
        self.assertEqual(expected, graph.friends_recommender('Red Imposter'))  # 8. Test reset vertices


if __name__ == '__main__':
    unittest.main()
