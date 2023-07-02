import argparse




def main():

    parsed_args = argparse.ArgumentParser("small tool to automate kpi generation")
    parsed_args.add_argument("--input-path", type=str, required = True, help="input path ...")
    parsed_args.add_argument("--output-path", type=str, required=True, help="output path ...")
    parsed_args.add_argument("--run-parallel", type=bool, required=False, help="flag for running in parallel ...")
    args = parsed_args.parse_args()

    print(args.input_path)
    print(args.output_path)
    print(args.run_parallel)

    # Example running commands:
    # python command_line_interface.py --input-path=" " --output-path=" " --run-parallel=True
