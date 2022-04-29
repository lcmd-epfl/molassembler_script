import scine_utilities as su
import scine_molassembler as masm
import numpy as np
from typing import Optional, Tuple, List, Iterable, Dict, Callable, Union


def cg_ensemble(mol: masm.Molecule, config: masm.dg.Configuration) -> List[np.array]:
    """Tries to generate an ensemble of conformations"""
    TRIAL_ENSEMBLE_SIZE = 50
    MIN_ENSEMBLE_SIZE = 1
    try:
        maybe_confs = masm.dg.generate_random_ensemble(mol, TRIAL_ENSEMBLE_SIZE, config)
        confs = []

        for conf in maybe_confs:
            if isinstance(conf, masm.dg.Error):
                continue

            error_norms = [
                np.linalg.norm(conf[i] - fixed_pos)
                for i, fixed_pos in config.fixed_positions
            ]

            if any(norm > 0.5 for norm in error_norms):
                continue

            for i, fixed_pos in config.fixed_positions:
                conf[i] = fixed_pos

            confs.append(conf)

        for conf in confs:
            if any(np.linalg.norm(r) > 1e3 for r in conf):
                masm.io.write("failed-conformer.svg", mol)
                masm.io.write("failed-conf.mol", mol, conf)
                raise RuntimeError("Bad CG result! See failed-conf* files.")

        if len(confs) < MIN_ENSEMBLE_SIZE:
            masm.io.write("failed-ensemble.svg", mol)
            raise RuntimeError(
                f"Generated only {len(confs)} confs: see failed-ensemble.svg"
            )

        return confs
    except RuntimeError as e:
        masm.io.write("problem-child.svg", mol)
        print("Encountered exception in CG. See problem-child.svg")
        raise e


dg_config = masm.dg.Configuration()
dg_config.partiality = masm.dg.Partiality.All
dg_config.refinement_step_limit = 10000
dg_config.refinement_gradient_target = 1e-5
dg_config.spatial_model_loosening = 1

filename = "pd_cl_xphos.xyz"
# We generate a Buchwald phosphine (XPhos) Pd complex
p_template = masm.io.experimental.from_smiles(
    "Cl[Pd]=P(c1ccccc1-c2c(C(C)(C))cc(C(C)(C))cc2(C(C)(C)))(C1CCCCC1)(C1CCCCC1)"
)
elements = p_template.graph.elements()
ensemble = cg_ensemble(p_template, dg_config)
converted = [su.AtomCollection(elements, conf) for conf in ensemble]

for j, conf in enumerate(converted):
    su.io.write(filename.replace(".xyz", f"-{j}.xyz"), conf)


filename = "pd_cl_johnphos.xyz"
# or JohnPhos
p_template = masm.io.experimental.from_smiles(
    "Cl[Pd]=P(c1ccccc1-c2ccccc2)(C(C)(C)C)(C(C)(C)C)"
)
elements = p_template.graph.elements()
ensemble = cg_ensemble(p_template, dg_config)
converted = [su.AtomCollection(elements, conf) for conf in ensemble]

for j, conf in enumerate(converted):
    su.io.write(filename.replace(".xyz", f"-{j}.xyz"), conf)
