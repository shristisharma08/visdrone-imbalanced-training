import argparse
import opts

def test_args():
    parser = opts.get_parser()
    args = parser.parse_args()  # Parse the arguments
    print(args)

if __name__ == '__main__':
    test_args()
