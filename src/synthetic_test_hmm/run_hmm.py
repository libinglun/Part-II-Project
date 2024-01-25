import argparse
from .main import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-iter', type=int, default=20)
    parser.add_argument('-state', type=str)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
