import scine_database as db
import scine_molassembler as masm
import scine_utilities as su
from time import sleep
from typing import NamedTuple, Any, List


class Options(NamedTuple):
    model: db.Model = db.Model("gfn2", "", "")


class JobOrder:
    GEOOPT = "scine_geometry_optimization"
    TSOPT = "scine_ts_optimization"
    GRAPH = "graph"
    BONDS = "scine_bond_orders"
    HESSIAN = "scine_hessian"


class Property:
    GIBBS = "gibbs_free_energy"
    PSEUDODIHEDRAL = "pseudodihedral"


class Graph:
    CBOR = "masm_cbor_graph"
    INDEX_MAP = "masm_idx_map"
    DEC_LIST = "masm_decision_list"

    @staticmethod
    def serialize_molecule(mol: masm.Molecule) -> str:
        serializer = masm.JsonSerialization
        cbor_format = serializer.BinaryFormat.CBOR
        serialization = serializer(mol)
        cbor_binary = serialization.to_binary(cbor_format)
        return serializer.base_64_encode(cbor_binary)

    @staticmethod
    def deserialize_cbor(cbor_str: str) -> masm.Molecule:
        serializer = masm.JsonSerialization
        cbor_binary = serializer.base_64_decode(cbor_str)
        cbor_format = serializer.BinaryFormat.CBOR
        serialization = serializer(cbor_binary, cbor_format)
        return serialization.to_molecule()

    @staticmethod
    def deserialize_molecules(structure: db.Structure) -> List[masm.Molecule]:
        cbors = structure.get_graph(Graph.CBOR)
        return [Graph.deserialize_cbor(m) for m in cbors.split(";")]

    @staticmethod
    def deserialize_molecule(struct: db.Structure) -> masm.Molecule:
        molecules = Graph.deserialize_molecules(struct)
        if len(molecules) != 1:
            su.io.write("split-mol.xyz", struct.get_atoms())
            msg = (
                "Multiple molecules stored for {}, but expected single "
                "molecule. See split-mol.xyz"
            )
            raise RuntimeError(msg.format(struct.id().string()))

        return molecules[0]


class Householder:
    options: Options
    manager: db.Manager
    calculations: db.Collection
    compounds: db.Collection
    structures: db.Collection
    properties: db.Collection

    def __init__(self, credentials=db.Credentials(), wipe=False):
        self.manager = db.Manager()
        self.manager.set_credentials(credentials)
        # self.manager.connect()
        if wipe:
            # self.manager.wipe()
            # self.manager.init()
            sleep(1.0)

        # self.calculations = self.manager.get_collection("calculations")
        # self.compounds = self.manager.get_collection("compounds")
        # self.structures = self.manager.get_collection("structures")
        # self.properties = self.manager.get_collection("properties")
        self.options = Options()

    def make_structure(
        self,
        atoms: su.AtomCollection,
        charge: int = 0,
        multiplicity: int = 1,
        label: db.Label = db.Label.MINIMUM_GUESS,
    ) -> db.Structure:
        """ Generate a structure in the database """
        structure = db.Structure()
        # structure.link(self.structures)
        # structure.create(atoms=atoms, charge=charge, multiplicity=multiplicity)
        # structure.set_label(label)
        return structure

    def make_compound(self, structures: List[db.Structure]) -> db.Compound:
        compound = db.Compound()
        # compound = db.Compound.make([s.id() for s in structures], self.compounds)
        # for s in structures:
        #    s.compound_id = compound.id()
        return compound

    def make_job(self, order: str, structure: db.Structure) -> db.Calculation:
        """ Generates a job, but does not trigger it """
        calculation = db.Calculation()
        # calculation.link(self.calculations)
        # calculation.create(self.options.model, db.Job(order), [structure.id()])
        return calculation

    def make_structure_optimization(self, structure: db.Structure) -> db.Calculation:
        """ Generates a geometry optimization, but does not trigger it """
        calculation = self.make_job(JobOrder.GEOOPT, structure)
        calculation.set_setting("convergence_max_iterations", 1000)
        calculation.set_setting("bfgs_use_trust_radius", True)
        calculation.set_setting("bfgs_trust_radius", 0.3)
        return calculation

    def make_ts_optimization(self, structure: db.Structure) -> db.Calculation:
        """ Generates a holding transition state optimization """
        calculation = self.make_job(JobOrder.TSOPT, structure)
        calculation.set_setting("convergence_max_iterations", 1000)
        calculation.set_setting("optimizer", "dimer")
        return calculation

    def fetch_scine_structure(self, obj: Any) -> db.Structure:
        assert obj is not None
        if isinstance(obj, db.ID):
            struct = db.Structure(obj)
        elif isinstance(obj, str):
            struct = db.Structure(db.ID(obj))
        else:
            try:
                # Try to handle mongoengine id
                struct = db.Structure(db.ID(str(obj)))
            except RuntimeError as e:
                msg = "Couldn't interpret {} of type {} into a structure"
                print(msg.format(obj, type(obj)))
                raise e

        struct.link(self.structures)
        assert struct.exists()
        return struct

    def add_pseudodihedral(
        self, structure: db.Structure, pseudodihedral: float
    ) -> db.Property:
        property = db.NumberProperty.make(
            Property.PSEUDODIHEDRAL, self.options.model, pseudodihedral, self.properties
        )
        structure.add_property(Property.PSEUDODIHEDRAL, property.id())
        return property

    def pseudodihedral(self, structure: db.Structure) -> float:
        if not structure.has_property(Property.PSEUDODIHEDRAL):
            prop_id = structure.get_property(Property.PSEUDODIHEDRAL)
            prop = self.properties.get_number_property(prop_id)
            return prop.data
        else:
            from substitute import Pseudodihedral

            mol = Graph.deserialize_molecule(structure)
            pseudodihedral = Pseudodihedral(mol)
            dihedral_value = pseudodihedral(structure.get_atoms())
            self.add_pseudodihedral(structure, dihedral_value)
            return dihedral_value
