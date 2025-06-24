import qiskit
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper

from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit_algorithms.minimum_eigensolvers import numpy_minimum_eigensolver
from qiskit_nature.second_q.circuit.library.ansatzes import UCCSD

from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock

from qiskit_algorithms.optimizers import SLSQP, ADAM
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

import concurrent.futures
import datetime
import time
import os

def run_vqe_for(basis, d, optimizer):
    # Define the molecule
    driver = PySCFDriver(
        atom    = f'Li .0 .0 .0; H .0 .0 {d}',
        basis   = basis,
        unit    = DistanceUnit.ANGSTROM
    )
    qmolecule = driver.run()

    # Get the electronic Hamiltonian considering only 4 spin orbitals
    transformer = ActiveSpaceTransformer(
        num_electrons = 2,
        num_spatial_orbitals = 2
    )
    molecule_reduced = transformer.transform(qmolecule)

    # Create the Fermionic operator and map it to qubits
    ferop = molecule_reduced.hamiltonian.second_q_op()
    mapper = ParityMapper(num_particles = molecule_reduced.num_particles) #Best type of mapper for this case
    qubit_op = mapper.map(ferop)
    
    # Solve analytically to get the ground state energy
    num_eigenval = 5
    solver = NumPyEigensolver(num_eigenval)
    result = solver.compute_eigenvalues(qubit_op)

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
        qubit_mapper         =  mapper,
        initial_state        = hf_circ,
        reps                 = 2
    )

    estimator = Estimator()

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

    return (d, vqe_result, result.eigenvalues[0])

f_time = datetime.datetime.now()

# Define the bases, mapper, optimizer, and output file
bases = [
    'sto3g',
    '321g',
    '631g',
    'ccpvtz'
]

# optimizer = SLSQP()
# opt_str = 'SLSQP'
optimizer = ADAM()
opt_str = 'ADAM'
mapper = 'parity'

dist_init = 0.01
dist_fin  = 3
steps     = 100
dist_array = np.linspace(dist_init, dist_fin, steps)

path = "/home/tommi/venvs/output"

for b in bases:
    tot = 0
    energy_dict = {}

    date = f'{f_time.day}-{f_time.month}-{f_time.year}_{f_time.hour}_{f_time.minute}_{f_time.second}'
    filename = f"results_{b}_{mapper}_{opt_str}_{date}.txt"
    full_path = os.path.join(path, filename)

    with open(full_path, 'w') as file:
        start = datetime.datetime.now()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_vqe_for, b, d, optimizer)
                for d in dist_array
            ]

            for future in concurrent.futures.as_completed(futures):
                d, vqe_res, exact_result = future.result()
                file.write(f"{d:.3f} Å - VQE = {vqe_res.total_energies[0]} (eigenvalue {vqe_res.eigenvalues[0]}) Ha, Classic = {exact_result} Ha\n")

        end = datetime.datetime.now()
        file.write(f"Total time: {(end - start).total_seconds():.2f} seconds")