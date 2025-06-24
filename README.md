# QCProject
Quantum Computing project: quantum chemistry simulations using qiskit nature.

This is the Quantum Computing course project. The objective is to demonstrate the effectiveness of quantum algorithm (VQE in this case).
Here we apply the VQE to the calculation of the groundstate of multiple molecules interacting at different distances, allowing to reproduce the Lennard-Jones binding potential. Both ideal and noisy simulations are performed. The only noise taken into consideration is the physical noise related to the gate application, i.e. the result of the gates applied in the VQE algorithm can lead to wrong result.
The optimizers used are ADAM, CG and SLSQP.
