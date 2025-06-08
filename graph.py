import igraph

for gen in range(100):
    for index in range(3):
        g = igraph.Graph.Read_Ncol(
            "C:\\Users\\Kush\\Documents\\My Files\\MY PROJECTS\\evolution_simulator\\images\\nnetGraph\\nnetText\\gen-"
            + str(gen).zfill(6)
            + "-index-"
            + str(index + 1)
            + ".txt",
            names=True,
            weights=True,
        )

        for v in g.vs:
            v["size"] = 35
            v["label"] = v["name"]
            if v["name"][0] == "S":
                v["color"] = "lightblue"
            elif v["name"][0] == "A":
                v["color"] = "lightgreen"
            else:
                v["color"] = "grey"

        for e in g.es:
            if e["weight"] < 0:
                e["color"] = "coral"
            elif e["weight"] == 0:
                e["color"] = "grey"
            else:
                e["color"] = "green"

            width = abs(e["weight"])
            e["width"] = 1 + 1.25 * (width / 8192.0)

        print(len(g.vs))

        bbox = (800, 800)
        layout = "fruchterman_reingold"

        igraph.plot(
            g,
            "C:\\Users\\Kush\\Documents\\My Files\\MY PROJECTS\\evolution_simulator\\images\\nnetGraph\\gen-"
            + str(gen).zfill(6)
            + "-index-"
            + str(index + 1)
            + ".svg",
            edge_curved=True,
            bbox=bbox,
            margin=64,
            layout=layout,
        )
