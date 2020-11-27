from argparse import ArgumentParser, SUPPRESS


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default='SUPPRESS')

    # custom command line input parameters
    args.add_argument("-r", "--resolution",
                      type=int, default=512, help="unproject voxel resolution N*N*N.")

    args.add_argument("-i", "--config",
                      type=str, default='../resources/camera.json', help="input config file.")
    
    args.add_argument("-o", "--output",
                      type=str, default='./', help="output visual hull mesh.")

    args.add_argument("-m", "--model",
                      type=str, default='../resources/visualhullMesh.ply', help="visual hull result for texture.")


    return parser
