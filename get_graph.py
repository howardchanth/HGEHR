from pathlib import Path

from construct_graph import GraphConstructor

from utils import ordered_yaml

import yaml


def main():
    opt_path = "construct_graph/MIMIC4.yml"
    opt_path = Path("./configs") / opt_path
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    graph_constructor = GraphConstructor(config)

    graph_constructor.load_mimic()
    graph_constructor.construct_graph()
    graph_constructor.set_tasks()
    graph_constructor.initialize_features()
    graph_constructor.save_graph()
    graph_constructor.save_mimic_dataset()


if __name__ == '__main__':
    main()
