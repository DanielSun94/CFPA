
def main():
    vertices_num_list = [20, 50, 100]
    edge_num_list = [1, 2, 3, 4, 5]
    idx = 0
    while idx < len(vertices_num_list) * len(edge_num_list):
        vertex_idx = idx // len(edge_num_list)
        edge_idx = idx % len(edge_num_list)
        vertex = vertices_num_list[vertex_idx]
        edge = edge_num_list[edge_idx]

        graph = Graph.Erdos_Renyi(n=vertex, m=edge*vertex, directed=True)
        if graph.is_dag():
            print('vertex: {}, edge: {}, acyclic ER graph pass'.format(vertex, edge*vertex))
            idx += 1


if __name__ == '__main__':
    main()
