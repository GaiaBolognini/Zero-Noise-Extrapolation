**Zero-Noise-Extrapolation**

Quantum Error Mitigation aim is to reduce the impact of noise that occurs in quantum computing: it is mainly based on performing different noisy simulations and infering the ideal noise-less expectation value. This technique can be efficiently implemented on near-term quantum devices as it does not require auxiliary qubits. On the other hand, it may be asymptotically unscalable: for longer circuits the cost of the computation would be high. 

In this work, a specific implementation of Quantum Error Mitigation is studied: the Zero-Noise_Exxtrapolation. It is based on two main steps:
**1.** Noise-Scaling: performing simulations at different levels of the processor noise, quantified by the scale factor($\lambda$). This step was performed through the Digital Noise Extrapolation or unitary folding technique, which requires only gate-level access to the system. Both the circuit folding and the gate-folding approaches have been implemented on Qiskit.
**2-** Extrapolation: at this point the problem is translated into a classical regression one. We can suppose it exists a model for the expectation values computed in the first step with respect to the scale factor. In this case, the Linear, the polynomial of order two and the exponential model were taken into account. 

At first, the implementation was checked in the case of the Randomized Benchmarking, widely used in this context. Then the tecnhique was used to mitigate the values obtained from the Grover's algorithm for differnt dimension of the database (2, 3 and 4 qubits). In the 3 qubit case, an improvement with respect to the unmitigated value was obtained for all the folding techniques and all the extrapolation models. In the 4 qubits case, the exponential trend is definitely more evident; the linear model does not allow to obtain better results with respect to the unmitigated values. Even in this case, no relevant difference was found in the folding technques. In the two qubits case, the behaviour of the scaling is strongly influenced by the Oracle, chosen to be unknown, which probably gives the most important contribution to the depth of the circuit, giving a plateau behaviour. 

The programs were written in Python using the Qiskit library. The simulations were run on the qasm simulator importing the noise model from a real backend, the ibmq_lima. 

