import os
import schnet.md as md
import numpy as np

if __name__ == '__main__':
    base_path = '/home/kschuett/bbdc/schnet/huziel_c20/results/bowl'

    ## model paths
    # best force
    mpath = 'SchNet_False_True_128_128_6_20.0_0.0001_1.0_split_bowl_0'
    # best energy
    force_mpath = 'SchNet_False_True_128_128_6_20.0_0.0001_1.0_split_bowl_0'
    # best compromise
    #mpath = 'schnett_True_True_128_128_6_20.0_False_0.0001_0.01_1.0_split20k_0'

    path = os.path.join(base_path, mpath)
#    mdpred = md.SchNetMD(path) # optional argument: nuclear charges

    # example prediction
    # positions: N_batch x N_atoms x 3
    pos = '1.5691,-0.6566,-0.9364,1.7669,0.6431,-0.472,0.4705,-0.6652,-1.7927,0.0116,0.6478,-1.8255,0.793,1.4673,-1.0284,-0.4874,-1.4818,-1.2157,-1.5635,-0.6572,-0.8952,-1.2694,0.649,-1.2767,-0.0023,-1.9618,-0.0072,-0.7698,-1.4532,1.0359,-1.7576,-0.638,0.4742,1.2878,-1.4503,0.1629,1.2896,-0.6595,1.3047,0.0115,-0.646,1.8533,1.583,0.6454,0.8984,0.4848,1.4383,1.1937,-0.5032,0.6469,1.7753,-1.6062,0.6715,0.9231,-1.2959,1.4891,-0.1655,-0.0102,1.9727,-0.0063'
    pos = np.array([float(p) for p in pos.split(',')])
    pos = pos.reshape((1, 20, 3))

    # array shapes:
    # energy: N_batch x 1
    # forces: N_batch x N_atoms x 3
 #   energy, forces = mdpred.get_energy_and_forces(pos)
  #  print(energy)
   # print(forces)


    force_path = os.path.join(base_path, force_mpath)
    # combined model
    mdpred = md.SchNetMD(path, force_path)
    energy, forces = mdpred.get_energy_and_forces(pos)
    print(energy)
    print(forces)

    #eq_pos = mdpred.relax(pos, eps=0.01, rate=5e-4)
    #print(eq_pos)