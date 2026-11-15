import argparse
from pathlib import Path


def component_die(line):
    parts = line.strip().split()
    if len(parts) < 3:
        return None

    suffix = parts[2].rsplit("_", 1)[-1].lower()
    if suffix in {"bottom", "upper"}:
        return suffix
    return None


def write_die_def(output_file, diearea, components):
    with open(output_file, "w") as outfile:
        outfile.write(diearea)
        outfile.write(f"COMPONENTS {len(components)} ;\n")
        outfile.writelines(components)
        outfile.write("END COMPONENTS\n")


def split_def_file(input_file, output_dir):
    output_dir = Path(output_dir)
    output_bottom = output_dir / "6_final_bottom.def"
    output_upper = output_dir / "6_final_upper.def"
    diearea = None
    components = {"bottom": [], "upper": []}
    in_components = False

    with open(input_file, "r") as infile:
        for line in infile:
            if diearea is None and line.startswith("DIEAREA"):
                diearea = line
                continue

            if line.startswith("COMPONENTS"):
                in_components = True
                continue

            if in_components:
                if line.startswith("END COMPONENTS"):
                    break

                die = component_die(line)
                if die is not None:
                    components[die].append(line)

    if diearea is None:
        raise ValueError(f"DIEAREA not found in {input_file}")

    write_die_def(output_bottom, diearea, components["bottom"])
    write_die_def(output_upper, diearea, components["upper"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEF File Splitter")
    parser.add_argument("-i", "--input", required=True, help="Input DEF file path")
    parser.add_argument("-o", "--output", default="output", help="Output directory")

    args = parser.parse_args()

    split_def_file(args.input, args.output)

    output_dir = Path(args.output)
    print(f"DEF: {output_dir / '6_final_bottom.def'} and "
          f"{output_dir / '6_final_upper.def'}")
