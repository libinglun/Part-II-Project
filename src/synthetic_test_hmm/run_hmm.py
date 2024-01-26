import argparse
from .main import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-iter', type=int, default=20)
    parser.add_argument('-state', type=str)
    parser.add_argument('-noise', type=float, default=0.5)
    parser.add_argument('-states', type=int, default=10)
    parser.add_argument('-obs', type=int, default=500)
    parser.add_argument('-size', type=int, default=50000)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
