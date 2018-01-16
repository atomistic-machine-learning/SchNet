# SchNet - a deep learning architecture for quantum chemistry
 
SchNet is a deep learning architecture that allows for spatially and chemically 
resolved insights into quantum-mechanical observables of atomistic systems.

Requirements:
- python 3.4
- ASE
- numpy
- tensorflow (>=1.0)

See the `scripts` folder for training and evaluation of SchNet 
model for predicting the total energy (U0) for the GDB-9 data set.

Install:

    python3 setup.py install

Download and convert QM9 data set:

    python3 load_qm9.py <qm9destination>

Train QM9 energy (U0) prediction:

    python3 train_energy_force.py <qm9destination>/qm9.db ./modeldir ./split50k.npz 
        --ntrain 50000 --nval 10000 --fit_energy --atomref <qm9destination>/atomref.npz


If you use SchNet in your research, please cite:

*K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*  
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017)

*K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
Quantum-chemical insights from deep tensor neural networks.*  
Nature Communications **8**. 13890 (2017)   
doi: [10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)
