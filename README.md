# SchNet - a deep learning architecture for quantum chemistry

_**Important: This package will not be further developed and supported. Please consider switching to our new pytorch-based package [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)!**_
 
SchNet is a deep learning architecture that allows for spatially and chemically 
resolved insights into quantum-mechanical observables of atomistic systems.

Requirements:
- python 3.4
- ASE
- numpy
- tensorflow (>=1.0)

See the `scripts` folder for training and evaluation of SchNet 
model for predicting the total energy (U0) for the GDB-9 data set.

## Install

    python3 setup.py install
    
    
## Examples

### QM9

Download and convert QM9 data set:

    python3 load_qm9.py <qm9destination>

Train QM9 energy (U0) prediction:

    python3 train_energy_force.py <qm9destination>/qm9.db ./modeldir ./split50k.npz 
        --ntrain 50000 --nval 10000 --fit_energy --atomref <qm9destination>/atomref.npz

### Potential energy surface

Predict force and energy for fullerene C20 configuration

    python scripts/example_md_predictor.py ./models/c20/ ./models/c20/C20.xyz
    
Relax geometry:

    python scripts/example_md_predictor.py ./models/c20/ ./models/c20/C20.xyz --relax
    
    
## References

If you use SchNet in your research, please cite:

*K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*  
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017)

*K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
Quantum-chemical insights from deep tensor neural networks.*  
Nature Communications **8**. 13890 (2017)   
doi: [10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)
