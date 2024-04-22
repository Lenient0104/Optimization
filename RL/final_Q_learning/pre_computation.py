class PreComputation:

    def find_all_routes(self, graph, start_edge, end_edge, path=[]):
        path = path + [start_edge]

        # If start is the same as end, we've found a path
        if start_edge == end_edge:
            return [path]

        if start_edge not in graph:
            return []

        paths = []
        for node in graph[start_edge]:
            if node not in path:
                newpaths = self.find_all_routes(graph, node, end_edge, path)
                for newpath in newpaths:
                    paths.append(newpath)

        return paths