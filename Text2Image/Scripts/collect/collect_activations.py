import os
import sys
from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Dualsteer_Code/Text2Image.Scripts.collect.cache_activations_runner import CacheActivationsRunner
from Dualsteer_Code/Text2Image.Scripts.train.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    CacheActivationsRunner(args).run()


if __name__ == "__main__":
    run()
