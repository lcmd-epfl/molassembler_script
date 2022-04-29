# molassembler_script
Using the Molassembler python API to generate an ensemble of TS guesses from a template.

This repository contains an exemplary script (`substitute.py`) demonstrating how Molassembler can be used to generate ensembles of TS structures from a template.
The provided snippet requires several [SCINE](https://scine.ethz.ch/download/) modules to run. [SCINE Molassembler](https://github.com/qcscine/molassembler) and [SCINE Utilities](https://github.com/qcscine/utilities) are a core part; SCINE Database dependencies also exist, but are unused in the current version because the geometry refinement and stationary point optimization can be performed separately from the generated 3D ensembles.

The script is pointed towards templates via the `templatepath` variable, and writes results to the `newpath` variable path. It systematically replaces the bare Cp ligand in the template with the (Cp-containing) molecule provided in `func_smiles`. It then attempts to generate `nconfs` conformers (50 by default) and saves them in `newpath`.

This is intended to serve as an example that can be adapted to use the Molassembler library for other (similar) purposes. An example of such adaptation is given by the `phosphine_conformer_generator.py`, which is significantly simpler and generates conformer samples for exemplary Buchwald phosphine Pd complexes.
