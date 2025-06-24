import qiskit
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import electronic_integrals
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper

from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit_algorithms.minimum_eigensolvers import numpy_minimum_eigensolver
from qiskit_nature.second_q.circuit.library.ansatzes import UCCSD

from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock

from qiskit_algorithms.optimizers import SLSQP, GradientDescent, ADAM
from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

import concurrent.futures
import datetime
import time

f_time = datetime.datetime.now()
basis = 'sto3g'
mapper = 'parity'
file = open(f'/home/tommi/venvs/files/results_{basis}_{mapper}_{f_time.hour}_{f_time.minute}_{f_time.second}_PARALL_TRIAL.txt', 'w')

dist_init = 0.01
dist_fin = 3
steps = 100
dist_array = np.linspace(dist_init, dist_fin, steps)

tot = 0

energy_dict = {}
for d in dist_array:
    # defining the molecular coordinates (aka spatial position in angstroms)
    driver = PySCFDriver(
        atom  = f'Li .0 .0 .0; H .0 .0 {d}', 
        basis = basis,
        unit  = DistanceUnit.ANGSTROM
    )
    qmolecule = driver.run()

    # collecting Hamiltonian related to the molecule
    electronic_energy: ElectronicEnergy = qmolecule.hamiltonian

    # calculating the energy configuration of the molecule: one_body = electrons are independent
    h1 = electronic_energy.electronic_integrals.one_body
    # calculating the energy configuration of the molecule: two_body = electron-electron interaction is considered
    h2 = electronic_energy.electronic_integrals.two_body

    # collecting nuclear repulsion energy
    nuclear_repulsion_energy = electronic_energy.nuclear_repulsion_energy

    ferm_op = electronic_energy.second_q_op()

    num_particles = sum(el for el in qmolecule.num_particles)

    transformer = ActiveSpaceTransformer(num_electrons = 2, num_spatial_orbitals = 2)
    molecule_reduced = transformer.transform(qmolecule)

    ferop = molecule_reduced.hamiltonian.second_q_op()

    mapper = ParityMapper(num_particles = molecule_reduced.num_particles)
    # mapper = JordanWignerMapper()
    qubit_op = mapper.map(ferop)


    num_eigenval = 5
    solver = NumPyEigensolver(num_eigenval)
    result = solver.compute_eigenvalues(qubit_op)


    hf_circ = HartreeFock(
        num_spatial_orbitals = molecule_reduced.num_spatial_orbitals,
        num_particles        = molecule_reduced.num_particles,
        qubit_mapper         = mapper
    )


    UCCSD_var_form = UCCSD(
        num_spatial_orbitals = molecule_reduced.num_spatial_orbitals,
        num_particles        = molecule_reduced.num_particles,
        qubit_mapper         = mapper,
        initial_state        = hf_circ,
        reps                 = 2
    )


    optimizers = [
        # GradientDescent(maxiter = 1000),
        SLSQP(),
        # ADAM()
    ]

    optimizers_str = [
        'GradientDescent',
        'SLSQP',
        'ADAM'
    ]

    estimator = Estimator()

    ### PARALLEL EXECUTION
    start = time.time()

    def run_vqe(opt_idx):
        opt = optimizers[opt_idx]
        vqe_solver = VQE(
            estimator = estimator,
            optimizer = opt,
            ansatz    = UCCSD_var_form
        )

        vqe_gs_solver = GroundStateEigensolver(
            qubit_mapper = mapper,
            solver       = vqe_solver
        )

        vqe_result = vqe_gs_solver.solve(molecule_reduced)

        return (optimizers_str[opt_idx], vqe_result.total_energies[0])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_vqe, idx) for idx in range(len(optimizers))]

        for future in concurrent.futures.as_completed(futures):
            key, value = future.result()
            energy_dict[key] = value

            file.write(f"{d}: {key} = {value} \n")

        # file.write('\n')

    end = time.time()
    tot += round(end - start, 3)
    
file.write(f'Total time: {tot} seconds')
file.close()
