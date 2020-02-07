import argparse

from gui import App

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Set checkpoint file path", type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    app = App(args.model)
    app.run()
