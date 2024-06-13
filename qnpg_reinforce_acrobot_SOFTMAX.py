from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import glob

pg=[]
qpg=[]

for np_name in glob.glob('*acrobot_softmax_NG - 0*.npy'):
    pg.append(np.load(np_name))

for np_name in glob.glob('*acrobot_softmax_NG - 1*.npy'):
    qpg.append(np.load(np_name))


pg_mean= np.array(pg).mean(axis=0)
qpg_mean= np.array(qpg).mean(axis=0)

window=10
smoothed_pg = [np.mean(pg_mean[i-window:i+1]) if i > window 
                    else np.mean(pg_mean[i-window:i+1]) for i in range(len(pg_mean))]
smoothed_qpg = [np.mean(qpg_mean[i-window:i+1]) if i > window 
                    else np.mean(qpg_mean[i-window:i+1]) for i in range(len(qpg_mean))]

#plt.figure(figsize=(12,8))
plt.plot(pg_mean,color="darkblue",alpha=0.1)
plt.plot(smoothed_pg,label="QPG",color="darkblue",alpha=0.7)

plt.plot(qpg_mean,color="purple",alpha=0.1)
plt.plot(smoothed_qpg,label="QNPG block diagonal",color="purple",alpha=0.7)

plt.legend(loc="lower right")#, ncol=3, prop={'size': 8})
plt.ylabel('Average Rewards')
plt.xlabel('Episodes')
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
plt.show()  