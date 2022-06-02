"""
 TFM - Deep Neuroevolution in Reconfigurable Hardware
"""

from argparse import ArgumentParser
ENVS_NAME = ["PongNoFrameskip-v4", "SpaceInvaders-v0", "Assault-v4"]

ENV_NAME = ENVS_NAME[0]

def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", default=ENV_NAME,
                        help="Environment name for AtariPy - OpenAI gym framework, default=" + ENV_NAME)
    parser.add_argument("-m", "--model", default=None, help="Initial weights for the first model")
    parser.add_argument("-s", "--seeds", default=None, help="Seeds for generate the initial weights for the first population")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-v", "--verbose", default=False, action='store_true',
                        help="Enable visualization of the game play")
    parser.add_argument("-d", "--debug", default=False, action='store_true', help="Debug mode")
    parser.add_argument("-r", "--record", default=False, action='store_true', help="Record ON")
    parser.add_argument("--vta", default=False, action='store_true', help="VTA mode")
    parser.add_argument("--gpu", default=False, action='store_true', help="GPU mode")
    args = parser.parse_args()

    if args.vta:

        from genetic_vta import GeneticVTA

        GeneticInstance = GeneticVTA(   environment=args.env, start_model_weights=args.model,
                                        output_dir=args.output, visualize=args.verbose,
                                        debug=args.debug, weight_seeds_path=args.seeds, record=args.record)
                                        
    elif args.gpu:

        from genetic_gpu import GeneticGPU

        GeneticInstance = GeneticGPU(   environment=args.env, start_model_weights=args.model,
                                        output_dir=args.output, visualize=args.verbose,
                                        debug=args.debug, weight_seeds_path=args.seeds, record=args.record)

    else:

        from genetic_cpu import GeneticCPU

        GeneticInstance = GeneticCPU(   environment=args.env, start_model_weights=args.model,
                                        output_dir=args.output, visualize=args.verbose,
                                        debug=args.debug, weight_seeds_path=args.seeds, record=args.record)

    GeneticInstance.run()


if __name__ == "__main__":
    main()