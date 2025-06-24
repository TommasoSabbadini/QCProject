import qiskit
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper, LogarithmicMapper

from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit_algorithms.minimum_eigensolvers import numpy_minimum_eigensolver
from qiskit_nature.second_q.circuit.library.ansatzes import UCCSD

from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock

from qiskit_algorithms.optimizers import SLSQP, ADAM, CG
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_algorithms import VQE
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

import concurrent.futures
import datetime
import time
import os

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

MAPPER_CLASSES = {
    'parity': ParityMapper,
    'jordan_wigner': JordanWignerMapper,
    'bravyi_kitaev': BravyiKitaevMapper,
    'logarithmic': LogarithmicMapper
}

def noise_model():
    # Define a basic depolarizing noise model
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.0001, 1)  # 0.01% error for 1-qubit gates
    error_2q = depolarizing_error(0.001, 2)   # 0.1% error for 2-qubit gates

    # Add to typical gates used in UCCSD
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    # Create AerSimulator with noise model
    noisy_backend = AerSimulator(
        noise_model          = noise_model,
        shots                = 1024, # Number of shots
        seed_simulator       = 42, # For reproducible results
        method               = 'density_matrix', # Good for small noisy systems
        max_parallel_threads = 1, # Avoid conflicts with multiprocessing
    )

    return noisy_backend

def physical_problem(atom1: str, atom2: str, distance: float, basis: str, num_el: int, num_spat_orb: int):
    """
    Defines the physical problem, i.e. the reduced molecule structure that will be used to create the qubit operator via the fermionic operator.

    **Inputs**:
        atoms composing the molecule, the distance between them and the basis

    **Output**:
        reduced molecule structure
    """
    # Define the molecule
    driver = PySCFDriver(
        atom    = f'{atom1} .0 .0 .0; {atom2} .0 .0 {distance}',
        basis   = basis,
        unit    = DistanceUnit.ANGSTROM
    )
    qmolecule = driver.run()

    # Get the electronic Hamiltonian considering only 4 spin orbitals
    transformer = ActiveSpaceTransformer(
        num_electrons        = num_el,
        num_spatial_orbitals = num_spat_orb
    )
    molecule_reduced = transformer.transform(qmolecule)

    return molecule_reduced

def get_mapper(mapper_name: str, num_particles: tuple[int, int]):
    mapper_class = MAPPER_CLASSES.get(mapper_name.lower())
    if mapper_class is None:
        raise ValueError(f"Unsupported mapper: {mapper_name}")
    
    return mapper_class(num_particles = num_particles)

def get_qubit_op(molecule_reduced, mapper_name: str):
    """
    Returns the qubit operator and mapper based on selected type.
    """

    fer_op = molecule_reduced.hamiltonian.second_q_op()
    mapper = get_mapper(mapper_name, molecule_reduced.num_particles)
    qubit_op = mapper.map(fer_op)

    return qubit_op, mapper

def classical_sol(qubit_op):
    """
    Solves numerically the problem.

    **Input**:
        qubit operator.

    **Output**:
        system eigenvalue.
    """
    # Solve analytically to get the ground state energy
    num_eigenval = 5
    solver = NumPyEigensolver(num_eigenval)
    result = solver.compute_eigenvalues(qubit_op)

    return result

def quantum_sol(molecule_reduced, mapper, optimizer, estimator: str):
    """
    Solves the problem using VQE algorithm. The initial state is defined by HartreeFock, the initial ansatz is defined using the UCSSD.

    **Input**:
        reduced molecule, mapper, optimizer.

    **Output**:
        system total energy.
    """
    # Initial state
    hf_circ = HartreeFock(
        num_spatial_orbitals = molecule_reduced.num_spatial_orbitals,
        num_particles        = molecule_reduced.num_particles,
        qubit_mapper         = mapper
    )

    # Ansatz for VQE
    UCCSD_var_form = UCCSD(
        num_spatial_orbitals = molecule_reduced.num_spatial_orbitals,
        num_particles        = molecule_reduced.num_particles,
        qubit_mapper         = mapper,
        initial_state        = hf_circ,
        reps                 = 2
    )

    vqe_solver = VQE(
        estimator = estimator,
        optimizer = optimizer,
        ansatz    = UCCSD_var_form
    ) 

    vqe_gs_solver = GroundStateEigensolver(
        qubit_mapper = mapper,
        solver       = vqe_solver
    )

    vqe_result = vqe_gs_solver.solve(molecule_reduced)

    return vqe_result

def run_for(atom1: str, atom2: str, basis: str, mapper_str: str, distance: float, optimizer, estimator):
    """
    Runs the simulation for every distance within the for cycle.

    **Input**:
        basis, distance, optimizer.

    **Output**:
        distance, quantum result (total system energy), classical result (numerically obtained eigenvalue).
    """
    molecule = physical_problem(atom1, atom2, distance, basis, 2, 2)
    qubit_operator, mapper = get_qubit_op(molecule, mapper_str)
    exact_energy = classical_sol(qubit_operator)
    vqe_result = quantum_sol(molecule, mapper, optimizer, estimator)

    return (distance, vqe_result, exact_energy)

f_time = datetime.datetime.now()

# Define the bases, mapper, optimizer, and output file
bases = [
    'sto3g',
    # '321g',
    # '631g',
    # 'ccpvtz'
]

# optimizer = SLSQP()
# opt_str = 'SLSQP'
# optimizer = CG()
# opt_str = 'CG'
optimizer = ADAM()
opt_str = 'ADAM'
mapper = 'parity'

atom1 = 'Li'
atom2 = 'H'

dist_init = 0.01
dist_fin  = 3
steps     = 100
dist_array = np.linspace(dist_init, dist_fin, steps)

noise = True
if noise:
    noisy_backend = noise_model()
    estimator = BackendEstimator(backend = noisy_backend)
else: estimator = Estimator()

path = "/home/tommi/venvs/output"

for b in bases:
    tot = 0
    energy_dict = {}

    date = f'{f_time.day}-{f_time.month}-{f_time.year}_{f_time.hour}_{f_time.minute}_{f_time.second}'
    filename = f"results{atom1}{atom2}_{b}_{mapper}_{opt_str}_{date}.txt"
    full_path = os.path.join(path, filename)

    with open(full_path, 'w') as file:
        start = datetime.datetime.now()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_for, atom1, atom2, b, mapper, d, optimizer, estimator) for d in dist_array
            ]

            for future in concurrent.futures.as_completed(futures):
                d, quantum_res, class_res = future.result()
                file.write(f"{d:.3f} Å - VQE = {quantum_res.total_energies[0]} (eigenvalue {quantum_res.eigenvalues[0]}) Ha, Classic = {class_res.eigenvalues[0]} Ha\n")

        end = datetime.datetime.now()
        file.write(f"Total time: {(end - start).total_seconds():.2f} seconds")