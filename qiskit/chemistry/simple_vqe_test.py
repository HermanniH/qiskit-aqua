
"""

Using this code you will be able to reduce the number of qubits by
finding underlying Z2 symmetries of the Hamiltonian.
The paper expaining the qubit reduction technique is:
by S. Bravyi et al. "Tapering off qubits to simulate fermionic Hamiltonians"
arXiv:1701.08213
This will drastically speed up all the simulations.

"""

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.components.optimizers import SLSQP, L_BFGS_B, SPSA, POWELL,COBYLA
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
import numpy as np
import matplotlib.pyplot as plt
# allows you to print the progress of simulations
import logging
from qiskit.aqua import set_qiskit_aqua_logging
from qiskit import IBMQ
import pickle
from qiskit.providers.aer import noise


provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-community', group='hackathon', project='tokyo-nov-2019')
#print(provider.backends())

#optimizer = POWELL(maxiter=None, maxfev=1000, disp=False, xtol=0.01, tol=None)
optimizer = COBYLA(maxiter=200, disp=False, rhobeg=1.0, tol=0.01)
# provides all information (can be too much text)
# set_qiskit_aqua_logging(logging.DEBUG)

# provides less information than DEBUG mode
set_qiskit_aqua_logging(logging.INFO)

#rs = np.linspace(0.45,1.6,5)
rs = [1.6]
energies = []
angles = []
machine = 'ibmqx2' #ibmqx2 #ibmq_vigo #'ibmq_ourense'#'ibmq_16_melbourne'



for r in rs:
    # set molecule and construct its Hamiltonian
    molecule = "H 0.000000 0.000000 0.000000;H 0.000000 0.000000 "+str(r)

    driver = PySCFDriver(atom=molecule,
                         unit=UnitsType.ANGSTROM,
                         charge=0,
                         spin=0,
                         basis='sto3g')

    qmolecule = driver.run()

    core = Hamiltonian(transformation=TransformationType.FULL,
                            qubit_mapping=QubitMappingType.JORDAN_WIGNER, # JORDAN
                            two_qubit_reduction=False,
                            freeze_core=True,
                            orbital_reduction=[])
    qubit_op, _ = core.run(qmolecule)

    # find the symmetries of the Hamiltonian
    #z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
    # tapered_ops = z2_symmetries.taper(qubit_op)
    # smallest_idx = 0  # Prior knowledge of which tapered_op has ground state
    # # or you can find the operator that has the ground state by diagonalising each operator
    # smallest_eig_value = 99999999999999
    # smallest_idx = -1
    # for idx in range(len(tapered_ops)):
    #     print('operator number: ', idx)
    #     ee = ExactEigensolver(tapered_ops[idx], k=1)
    #     curr_value = ee.run()['energy']
    #     if curr_value < smallest_eig_value:
    #         smallest_eig_value = curr_value
    #         smallest_idx = idx
    # print('Operator number: ', smallest_idx, ' contains the ground state.')
    #
    # # the tapered Hamiltonian operator
    # the_tapered_op = tapered_ops[smallest_idx]

    the_tapered_op= qubit_op
    # optimizers
    # optimizer = SLSQP(maxiter=1000)
    #optimizer = L_BFGS_B(maxiter=1000)

    # initial state
    # init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
    #                          num_orbitals=core._molecule_info['num_orbitals'],
    #                          qubit_mapping=core._qubit_mapping,
    #                          two_qubit_reduction=core._two_qubit_reduction,
    #                          num_particles=core._molecule_info['num_particles'],
    #                          sq_list=the_tapered_op.z2_symmetries.sq_list)
    init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
                             num_orbitals=core._molecule_info['num_orbitals'],
                             qubit_mapping=core._qubit_mapping,
                             two_qubit_reduction=core._two_qubit_reduction,
                             num_particles=core._molecule_info['num_particles'],)
                             # sq_list=the_tapered_op.z2_symmetries.sq_list)
    # UCCSD Ansatz
    # var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
    #                  num_orbitals=core._molecule_info['num_orbitals'],
    #                  num_particles=core._molecule_info['num_particles'],
    #                  active_occupied=None, active_unoccupied=None,
    #                  initial_state=init_state,
    #                  qubit_mapping=core._qubit_mapping,
    #                  two_qubit_reduction=core._two_qubit_reduction,
    #                  num_time_slices=1,
    #                  z2_symmetries=the_tapered_op.z2_symmetries,
    #                  shallow_circuit_concat=False)
    var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
                     num_orbitals=core._molecule_info['num_orbitals'],
                     num_particles=core._molecule_info['num_particles'],
                     active_occupied=None, active_unoccupied=None,
                     initial_state=init_state,
                     qubit_mapping=core._qubit_mapping,
                     two_qubit_reduction=core._two_qubit_reduction,
                     num_time_slices=1,
                     shallow_circuit_concat=False)


    # set up VQE
    algo = VQE(the_tapered_op, var_form, optimizer)


    # For simulating classically with noise
    # Choose the backend (use Aer instead of BasicAer)
    # backend = Aer.get_backend('qasm_simulator')
    # provider = IBMQ.load_account()
    # provider = IBMQ.get_provider(hub='ibm-q-community', group='hackathon', project='tokyo-nov-2019')
    # backend_sim = provider.get_backend(machine)
    # backendConfig = backend_sim.configuration()
    # properties = backend_sim.properties()
    # coupling_map = backendConfig.coupling_map
    # noise_model = noise.device.basic_device_noise_model(properties)
    # basis_gates = noise_model.basis_gates
    #quantum_instance = QuantumInstance(backend=backend, optimization_level=0, shots = 3000,noise_model=noise_model,basis_gates=basis_gates)#, initial_layout=[3,4])


    #For running on real hardware
    #backend = IBMQ.get_provider().get_backend('machine')
    backend = provider.get_backend(machine)
    quantum_instance = QuantumInstance(backend=backend, optimization_level=3, shots = 3000)

    # run the algorithm
    algo_result = algo.run(quantum_instance)

    # get the results
    _, result = core.process_algorithm_result(algo_result)
    print(result)

    energy = result['energy']
    energies.append(energy)
    angles.append(result['algorithm_retvals']['opt_params'])
outfile = open('real_H2_16.pcl','wb')
pickle.dump([rs,energies,angles],outfile)
outfile.close()

plt.plot(rs,energies)
plt.show()