import scine_utilities as su
import scine_molassembler as masm
import scine_database as db

import sys
import numpy as np
import random
import glob
from typing import Optional, Tuple, List, Iterable, Dict, Callable, Union
from functools import partial
from collections import defaultdict


from householding import Householder

"""

Overview

- Find Cp in each structure
- Substitute the Cp with something else
- Generate conformers
  - First coordination sphere (excluding Cp and R) fixed, rest free
  - Enough to sample Cp rotational freedom

"""

# Main variables to set up the structure generation
templatepath = "templates/UR.xyz"
newpath = "UR.xyz"
nconfs = 50
CENTRAL_ELEMENT = su.ElementType.Rh

# Approximate coordinates of the metal center in the template, useful to define clockwise/counterclockwise angles
ref_coords = [-1.5, -2, 0]
MaybePredicate = Optional[Callable[[su.AtomCollection], bool]]

# Indices to define either an angle (3 indices) or a dihedral (4 indices) to keep track of axial chirality. Can be None if not needed.
idxs = None  # e.g. [31, 32, 33] # [6, 17, 18, 29]
# Direction can be True or False to switch the direction (axial chirality) of the attachment of the Cp
direction = True


# The SMILES in this string will be used to replace Cp in the template. Must contain a Cp-like structure.
func_smiles = (
    "c12c[cH-]cc1[C@H](C)[C@H]3OC(c1ccccc1)(c1ccccc1)O[C@@H]3[C@@H]2(C)"  # Catalyst 1
)

# Options to functionalize other positions of the Cp ring, in case the SMILES is not convenient
tip_smiles = None
tip_atom = su.ElementType.C
cindex_choice = 2


def is_flat_carbon(idx: int, mol: masm.Molecule) -> bool:
    if mol.graph.element_type(idx) != su.ElementType.C:
        return False

    permutator = mol.stereopermutators.option(idx)
    if not permutator:
        return False

    Shape = masm.shapes.Shape
    flat_shapes = [Shape.Bent, Shape.EquilateralTriangle]
    return permutator.shape in flat_shapes


def find_cyclopentadienyl(mol: masm.Molecule) -> Optional[List[int]]:
    cp_pattern = masm.io.experimental.from_smiles("[C]1[C][C][C][C]1")
    matches = masm.subgraphs.complete(cp_pattern, mol)
    if len(matches) == 0:
        return None

    # There are likely to be multiple matches since the pattern is rotationally
    # symmetric, so just pick the first one that works
    for match in matches:
        atoms = [j for i, j in match.left]
        if all(map(partial(is_flat_carbon, mol=mol), atoms)):
            return atoms

    return None


class StableIndexManager:
    """
    Allows you to keep using your atom indices while you're deleting atoms in a
    molecule (which principally invalidates atom indices)

    >>> manager = StableIndexManager()
    >>> # Let's use partial, which doesn't raise on reusing deleted indices
    >>> [manager.partial(i) for i in range(8)]
    [0, 1, 2, 3, 4, 5, 6, 7]
    >>> manager.delete(4)
    >>> [manager.partial(i) for i in range(8)]
    [0, 1, 2, 3, None, 4, 5, 6]
    >>> manager.delete(2)
    >>> [manager.partial(i) for i in range(8)]
    [0, 1, None, 2, None, 3, 4, 5]
    """

    deleted_indices: List[int]

    def __init__(self):
        self.deleted_indices = []

    def __call__(self, index: int) -> int:
        result = index
        for i in self.deleted_indices:
            if i == index:
                raise RuntimeError("This atom was deleted!")

            if result > i:
                result -= 1

        return result

    def partial(self, index: int) -> Optional[int]:
        try:
            return self(index)
        except RuntimeError:
            return None

    def delete(self, index: int):
        self.deleted_indices.append(index)
        self.deleted_indices.sort(reverse=True)


def find_cyclopentadienyl_site(mol: masm.Molecule) -> Optional[Tuple[int, int]]:
    """Returns atom index and site index pair"""
    flat_carbon = partial(is_flat_carbon, mol=mol)

    for permutator in mol.stereopermutators.atom_stereopermutators():
        for site_idx, site in enumerate(permutator.ranking.sites):
            if len(site) == 5 and all(map(flat_carbon, site)):
                return (permutator.placement, site_idx)

    return None


class Pseudodihedral:
    """Class for calculating rotation of a substituted cyclopentadienyl"""

    Sequence = List[Union[int, List[int]]]
    definition: Sequence

    def __init__(self, mol: masm.Molecule):
        """Find a working definition of the pseudodihedral"""
        maybe_atom_site_pair = find_cyclopentadienyl_site(mol)
        if not maybe_atom_site_pair:
            raise RuntimeError("Couldn't find Cp for pseudodihedral")

        second, cyclopentadienyl_site = maybe_atom_site_pair
        ranking = mol.stereopermutators[second].ranking
        sites = ranking.sites
        ranked_substituents = ranking.ranked_substituents

        # Select the first fixture as the lowest ranked non-haptic site atom
        lowest_ranked_nonhaptic_sites = [
            equivalent_sites[0]
            for equivalent_sites in ranking.ranked_sites
            if len(equivalent_sites) == 1 and len(sites[equivalent_sites[0]]) == 1
        ]
        assert len(lowest_ranked_nonhaptic_sites) > 0
        first = sites[lowest_ranked_nonhaptic_sites[0]][0]
        third = sites[cyclopentadienyl_site]

        # Select the highest ranked unique substituent in the haptic site
        def atom_rank(atom: int) -> int:
            for i, equivalent_atom_list in enumerate(ranked_substituents):
                if atom in equivalent_atom_list:
                    return i

            raise IndexError(f"No atom {atom} in ranked substituents")

        ranks: Dict[int, List[int]] = defaultdict(list)
        for atom in third:
            ranks[atom_rank(atom)].append(atom)

        rank_descending = sorted(ranks.items(), reverse=True)
        uniques = [atoms[0] for rank, atoms in rank_descending if len(atoms) == 1]
        if len(uniques) == 0:
            raise RuntimeError("No uniquely ranked atom in haptic site")

        fourth = uniques[0]
        self.definition = [first, second, third, fourth]

    def __call__(self, ac: su.AtomCollection) -> float:
        """Returns a calculated dihedral in radians in [-pi, pi)"""

        def average(ac: su.AtomCollection, x: Union[int, List[int]]) -> np.ndarray:
            if isinstance(x, int):
                return ac.get_position(x)

            return sum([ac.get_position(i) for i in x]) / len(x)

        m = np.array([average(ac, atom_or_list) for atom_or_list in self.definition])
        return Pseudodihedral.dihedral_angle(m)

    @staticmethod
    def dihedral_angle(p: np.ndarray) -> float:
        """Calculates the dihedral between four points"""
        b0 = -1.0 * (p[1] - p[0])
        b1 = p[2] - p[1]
        b2 = p[3] - p[2]

        b1 /= np.linalg.norm(b1)

        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)

    @staticmethod
    def normal_angle(p: np.ndarray, ref: np.ndarray) -> float:
        """Calculates the angle between three points"""
        b0 = p[0] - p[1]
        b1 = p[2] - p[1]
        b0 /= np.linalg.norm(b0)
        b1 /= np.linalg.norm(b1)
        x = np.dot(b0, b1)
        w = ref - p[1]

        w /= np.linalg.norm(w)
        y = np.dot(w, np.cross(b0, b1))
        a = np.arctan2(y, x)
        return a


def fixed_atoms(mol: masm.Molecule) -> List[int]:
    """Makes a list of the central metal and all its immediate adjacents"""
    metal = mol.graph.elements().index(CENTRAL_ELEMENT)
    first_shell = list(mol.graph.adjacents(metal))
    second_shell = []
    third_shell = []
    for idx in first_shell:
        linked = mol.graph.adjacents(idx)
        second_shell.extend(linked)
    shells = first_shell + second_shell + third_shell + [metal]
    return list(set(shells))


class TrackedSubstitution:
    """Molecule and atom index map for tracking index changes"""

    mol: masm.Molecule
    positions: Dict[int, np.ndarray]
    ligand_index_map: Dict[int, int]

    def __init__(
        self,
        mol: masm.Molecule,
        positions: Dict[int, np.ndarray],
        ligand_index_map: Dict[int, int],
    ):
        self.mol = mol
        self.positions = positions
        self.ligand_index_map = ligand_index_map

    @property
    def fixed_first_shell(self) -> List[int]:
        graph_fixed = frozenset(fixed_atoms(self.mol))
        return list(frozenset(self.positions.keys()) & graph_fixed)


def substitute_cp(
    mol: masm.Molecule, ligand: masm.Molecule, ac: su.AtomCollection
) -> Optional[TrackedSubstitution]:
    """Replace a cyclopentadienyl by a substituted form"""
    # Generate ligand molecule and find flat carbon five ring
    ligand_cp_atoms = find_cyclopentadienyl(ligand)
    if ligand_cp_atoms is None:
        print("Couldn't find Cp in ligand!")
        return None

    # Find cyclopentadienyl in the structure
    cp_site_pair = find_cyclopentadienyl_site(mol)
    if cp_site_pair is None:
        randnum = random.randint(0, 9999)
        fname = f"no-cp-{randnum}.svg"
        masm.io.write(fname, mol)
        print(f"Couldn't find Cp in molecule. Wrote {fname}.")
        return None

    # Substitute Cp ligand
    anchor_element = mol.graph.element_type(cp_site_pair[0])
    cleaved = masm.editing.cleave(mol, cp_site_pair)
    split = cleaved.first  # cleaved.second, the cp, is discarded unused
    component_map = cleaved.component_map
    partial_positions = {
        after: ac.positions[before]
        for before, (component, after) in enumerate(component_map)
        if component == 0
    }

    central = split.graph.atoms_of_element(anchor_element)[0]
    complete = masm.editing.add_ligand(split, ligand, central, ligand_cp_atoms)

    # Tip and other substitutions if desired
    if tip_smiles is not None:
        adj_h = []
        g = complete.graph
        ligand_cp_atoms = find_cyclopentadienyl(complete)
        for idx, ligand_cp_atom in enumerate(ligand_cp_atoms[:]):
            if idx == cindex_choice:
                for adj in g.adjacents(ligand_cp_atoms[idx]):
                    if (
                        g.element_type(adj) == su.ElementType.C
                        and adj not in ligand_cp_atoms[:]
                    ):
                        continue
                    if g.element_type(adj) == su.ElementType.H and g.can_remove(adj):
                        tip = Ligand(tip_smiles, "Tip")
                        tipg = tip.mol.graph
                        for catm in tipg.atoms_of_element(tip_atom):
                            counter = 0
                            for cadj in tipg.adjacents(catm):
                                if tipg.element_type(cadj) == su.ElementType.C:
                                    counter += 1
                                elif tipg.element_type(
                                    cadj
                                ) == su.ElementType.H and tipg.can_remove(cadj):
                                    hindex = cadj
                            if counter == 3:
                                cindex = catm
                                break
                        try:
                            complete = masm.editing.substitute(
                                complete,
                                tip.mol,
                                masm.BondIndex(ligand_cp_atoms[idx], adj),
                                masm.BondIndex(cindex, hindex),
                            )
                        except IndexError as m:
                            raise IndexError(m)
    ligand_index_map = {i: split.graph.V + i for i in ligand.graph.atoms()}
    return TrackedSubstitution(complete, partial_positions, ligand_index_map)


def interpret(ac: su.AtomCollection) -> masm.Molecule:
    """Interprets an atom collection, no index shuffles"""
    bo = su.BondDetector.detect_bonds(ac)
    discretization = masm.interpret.BondDiscretization.Binary
    interpreted = masm.interpret.molecules(ac, bo, discretization)
    assert len(interpreted.molecules) == 1
    mol = interpreted.molecules[0]
    return mol


def cg_ensemble(mol: masm.Molecule, config: masm.dg.Configuration) -> List[np.array]:
    """Tries to generate an ensemble of conformations"""
    TRIAL_ENSEMBLE_SIZE = nconfs
    MIN_ENSEMBLE_SIZE = 1
    try:
        maybe_confs = masm.dg.generate_random_ensemble(mol, TRIAL_ENSEMBLE_SIZE, config)
        confs = []

        for conf in maybe_confs:
            if isinstance(conf, masm.dg.Error):
                continue

            # Temporary fix for soft coordinate fixing in molassembler (#193):
            # Overwrite fixed positions if the quaternion fit is reasonable
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

        # masm.io.write("check.svg", mol)
        return confs
    except RuntimeError as e:
        masm.io.write("problem-child.svg", mol)
        print("Encountered exception in CG. See problem-child.svg")
        raise e


def generate_conformer_ensemble(
    substitution: TrackedSubstitution, maybe_predicate: MaybePredicate
) -> List[su.AtomCollection]:
    """Generate a set of conformers with fixed positions"""
    dg_config = masm.dg.Configuration()
    graph_fixed = frozenset(fixed_atoms(substitution.mol))
    dg_config.fixed_positions = [
        (i, pos) for i, pos in substitution.positions.items() if i in graph_fixed
    ]
    try:
        ensemble = cg_ensemble(substitution.mol, dg_config)
        elements = substitution.mol.graph.elements()
        converted = [su.AtomCollection(elements, conf) for conf in ensemble]
        if not maybe_predicate:
            return converted

        reduced = [ac for ac in converted if maybe_predicate(ac)]
        print(f"Before predicate: {len(converted)}, after: {len(reduced)}")
        return reduced
    except RuntimeError as e:
        print(f"Ensemble generation exception: {e}")
        return []


def iter_structures() -> Iterable[Tuple[str, su.AtomCollection]]:
    """Iterate through template TS variations"""
    prefix = templatepath
    filenames = glob.glob(prefix)
    for filename in filenames:
        ac, _ = su.io.read(filename)
        yield (filename, ac)


class Ligand:
    mol: masm.Molecule
    name: str
    cg_predicate: MaybePredicate = None

    def __init__(self, smiles: str, name: str, pred: MaybePredicate = None):
        try:
            self.mol = masm.io.experimental.from_smiles(smiles)
        except RuntimeError as m:
            print(smiles)
            raise RuntimeError(m)
        self.name = name
        self.predicate = pred


class DihedralPredicate:
    sequence: List[int]
    positive: bool

    def __init__(self, sequence: List[int], positive: bool):
        self.sequence = sequence
        self.positive = positive

    def __call__(self, ac: su.AtomCollection) -> bool:
        positions = np.array([ac.get_position(i) for i in self.sequence])
        dihedral = Pseudodihedral.dihedral_angle(positions)
        return dihedral > 0 if self.positive else dihedral < 0


class AnglePredicate:
    sequence: List[int]
    positive: bool

    def __init__(self, sequence: List[int], positive: bool):
        self.sequence = sequence
        self.positive = positive

    def __call__(self, ac: su.AtomCollection) -> bool:
        positions = np.array([ac.get_position(i) for i in self.sequence])
        angle = Pseudodihedral.normal_angle(positions, np.array(ref_coords))
        return angle > 0 if self.positive else angle < 0


def iter_ligands() -> Iterable[Ligand]:
    """Cp ligands in SMILES format"""
    if idxs is None:
        cp_replacements = [Ligand(func_smiles, "ModifiedLigand")]
    elif direction is not None and len(idxs) == 4:
        cp_replacements = [
            Ligand(
                func_smiles,
                "ModifiedLigand",
                DihedralPredicate([idxs[3], idxs[2], idxs[1], idxs[0]], direction),
            )
        ]
    elif direction is not None and len(idxs) == 3:
        cp_replacements = [
            Ligand(
                func_smiles,
                "ModifiedLigand",
                AnglePredicate([idxs[0], idxs[1], idxs[2]], direction),
            )
        ]
    for ligand in cp_replacements:
        yield ligand


def peek():
    for filename, ac in iter_structures():
        mol = interpret(ac)
        for i, ligand in enumerate(iter_ligands()):
            substituted = substitute_cp(mol, ligand.mol, ac)
            if substituted is None:
                continue

            if isinstance(ligand.predicate, DihedralPredicate):
                ligand.predicate.sequence = [
                    substituted.ligand_index_map[i] for i in ligand.predicate.sequence
                ]

            ensemble = generate_conformer_ensemble(substituted, ligand.predicate)
            for j, conf in enumerate(ensemble):
                su.io.write(filename.replace(".xyz", f"-{i}-{j}.xyz"), conf)

            sys.exit(0)


# Unused functions meant to perform the optimization fully using SCINE
# def xtb_minimize(
#    substitution: TrackedSubstitution, ac: su.AtomCollection
# ) -> su.AtomCollection:
#    import scine_xtb
#
#    manager = su.core.ModuleManager()
#    calculator = manager.get("calculator", "GFN2")
#    if not calculator:
#        raise RuntimeError("No calculator for XTB")
#
#    assert isinstance(calculator, su.core.Calculator)
#    calculator.structure = ac
#    log = su.core.Log()
#
#    optimizer = su.Optimizer.SteepestDescent
#    settings = su.geometry_optimization_settings(calculator, optimizer)
#    settings.update(
#        {
#            "convergence_max_iterations": 1000,
#            "geoopt_transform_coordinates": False,
#            "geoopt_coordinate_system": "cartesian",
#            "geoopt_constrained_atoms": substitution.fixed_first_shell,
#        }
#    )
#    return su.geometry_optimize(calculator, log, optimizer, settings=settings)
#
#
# def make_tsopt(
#    structure: db.Structure, substituted: TrackedSubstitution, householder: Householder
# ) -> db.Calculation:
#    TSOPT_SETTINGS = {
#        "convergence_max_iterations": 1000,
#        "geoopt_coordinate_system": "cartesian",
#        "geoopt_constrained_atoms": substituted.fixed_first_shell,
#        "optimizer": "dimer",
#    }
#    calculation = householder.make_ts_optimization(structure)
#    for k, v in TSOPT_SETTINGS.items():
#        calculation.set_setting(k, v)
#    return calculation
#
#
# def make_geoopt(
#    structure: db.Structure, substituted: TrackedSubstitution, householder: Householder
# ) -> db.Calculation:
#    GEOOPT_SETTINGS = {
#        "convergence_max_iterations": 1000,
#        "bfgs_use_trust_radius": True,
#        "bfgs_trust_radius": 0.3,
#        "geoopt_coordinate_system": "bofill",
#        "geoopt_constrained_atoms": substituted.fixed_first_shell,
#    }
#    calculation = householder.make_structure_optimization(structure)
#    for k, v in GEOOPT_SETTINGS.items():
#        calculation.set_setting(k, v)
#    return calculation


def generate_gaussian_inputfile(filename):
    ginput_filename = filename.replace(".xyz", ".com")
    with open(filename, "r") as f:
        coords = "".join(f.readlines()[2:])
    chkname = filename.replace(".xyz", ".chk")

    before = f"""%chk={chkname}
# opt=(calcfc, ts, noeigentest,maxstep=20) freq=noraman external=\"xtb-gaussian --alpb Methanol\" IOP(1/18=20) IOP(1/6=500)

{filename}

0 1
"""

    after = f"""
--Link 1--
%chk={chkname}
%mem=25000mb
%nprocshared=8
# opt=(calfc, ts, noeigentest,maxstep=20) B3PW91/def2svp empiricaldispersion=gd3bj freq geom=allcheck

--Link 1--
%mem=25000mb
%nprocshared=8
%chk={chkname}
# B3P91/def2TZVP empiricaldispersion=gd3bj guess=read geom=allcheck scrf=(smd,solvent=ethanol)
"""

    with open(ginput_filename, "w+") as f:
        f.write(before)
        f.write(coords)
        f.write(after)


def initialize(householder: Householder):
    """Generate the initial set of structures and calculations"""
    for filename, ac in iter_structures():
        is_ts = filename.startswith("ts")
        label = db.Label.TS_GUESS if is_ts else db.Label.MINIMUM_GUESS
        mol = interpret(ac)
        for i, ligand in enumerate(iter_ligands()):
            substituted = substitute_cp(mol, ligand.mol, ac)
            if substituted is None:
                continue

            if isinstance(ligand.predicate, DihedralPredicate):
                ligand.predicate.sequence = [
                    substituted.ligand_index_map[i] for i in ligand.predicate.sequence
                ]

            ensemble = generate_conformer_ensemble(substituted, ligand.predicate)
            structures = [householder.make_structure(c, label=label) for c in ensemble]

            # Write files?
            if True:
                for j, conf in enumerate(ensemble):
                    destination = filename.replace(f"{templatepath}", f"{newpath}")
                    output_filename = destination.replace(".xyz", f"-{j}.xyz")
                    su.io.write(output_filename, conf)
                    generate_gaussian_inputfile(output_filename)

            # Link up a compound of all these structures
            householder.make_compound(structures)

            # Set up optimizations if desired (requires additional SCINE modules)
            if is_ts:
                # factory = make_tsopt
                pass
            else:
                # factory = make_geoopt
                pass
            for structure in structures:
                # calculation = factory(structure, substituted, householder)
                # calculation.set_status(db.Status.NEW)
                pass


if __name__ == "__main__":
    creds = db.Credentials()
    creds.database_name = "Molassembler"
    householder = Householder(creds, wipe=True)
    initialize(householder)
