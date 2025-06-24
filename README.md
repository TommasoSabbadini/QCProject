# QCProject
Quantum Computing project: quantum chemistry simulations using qiskit nature.

This is the Quantum Computing course project developed by two people that first encountered quantum chemistry developing this codes. The objective is to demonstrate the effectiveness of quantum algorithms like VQE. Comparisons between the quantum and classical computation are present in data analysis files not uploaded in this repository.

Here we apply the VQE for the calculation of the ground state of multiple molecules interacting at different distances, allowing to reproduce the Lennard-Jones binding potential. Both ideal and noisy simulations are performed. The only noise taken into consideration is the physical noise related to the gate application, i.e. the result of gates applied in the VQE algorithm can produce wrong outcome.

The optimizers used are ADAM, CG and SLSQP.
