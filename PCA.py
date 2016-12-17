
# coding: utf-8

# In[ ]:

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
#read data
train = np.genfromtxt ('digits/train.csv', delimiter=",")                  #784 by 5000
trainlabels = np.genfromtxt ('digits/trainlabels.csv', delimiter=",")      #5000 by 1

[n,m]=np.shape(train)  # number of pixels and number of training data(n =784 and m =5000 in this case)
#nomrmalize  data
normal = np.dot(train.T,train)
normal = np.sqrt(np.diag(normal))
train_new = train / np.matlib.repmat(normal,n,1)
train_new = train_new.T

# number of Principal Components to save
nPC = 6
PCV = np.zeros((n,nPC))
# The mean substracted is across each dimension
mean = np.mean(train_new,axis=0)
mean = np.matlib.repmat(mean,m,1)
train_new = train_new - mean
Cov = np.cov(train_new.T)

# Solve an ordinary or generalized eigenvalue problem
# for a complex Hermitian or real symmetric matrix.
eigen_val, eigen_vec = np.linalg.eigh(Cov)

# sorting the eigenvalues in descending order
idx = np.argsort(eigen_val)
idx = idx[::-1]
# sorting eigenvectors according to the sorted eigenvalues
eigen_vec = eigen_vec[:,idx]
# sorting eigenvalues
eigen_val = eigen_val[idx]

# save only the most significant eigen vectors
PCV[:,:nPC] = eigen_vec[:,:nPC]

# apply transformation
FinalData = train_new.dot(PCV)

# find indexes of data for each digit
zeroData  = (trainlabels==0).nonzero()#red
twoData   = (trainlabels==2).nonzero()#green
fourData  = (trainlabels==4).nonzero()#m
sevenData = (trainlabels==7).nonzero()#yellow
oneData   = (trainlabels==1).nonzero()#blue


# figure #first second and third PC
fig = plt.figure()
ax = fig.gca(projection = '3d')

# plot zeros
xcomp = FinalData[zeroData,0].flatten()
ycomp = FinalData[zeroData,1].flatten()
zcomp = FinalData[zeroData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,0].flatten()
ycomp = FinalData[twoData,1].flatten()
zcomp = FinalData[twoData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'g.')
# plot fours
xcomp = FinalData[fourData,0].flatten()
ycomp = FinalData[fourData,1].flatten()
zcomp = FinalData[fourData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,0].flatten()
ycomp = FinalData[sevenData,1].flatten()
zcomp = FinalData[sevenData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[oneData,0].flatten()
ycomp = FinalData[oneData,1].flatten()
zcomp = FinalData[oneData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('1st, 2nd and 3rd principal components')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
#set legend for different colour data point
rpatch = mpatches.Patch(color='r', label='0')
gpatch = mpatches.Patch(color='g', label='2')
mpatch = mpatches.Patch(color='m', label='4')
ypatch = mpatches.Patch(color='y', label='7')
bpatch = mpatches.Patch(color='b', label='1')
ax.legend(handles=[rpatch, gpatch, mpatch,ypatch,bpatch], fontsize = 'large',loc = 'upper left')
plt.show()


# figure #first second and fourth PC
fig = plt.figure()
ax = fig.gca(projection = '3d')
# plot zeros
xcomp = FinalData[zeroData,0].flatten()
ycomp = FinalData[zeroData,1].flatten()
zcomp = FinalData[zeroData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,0].flatten()
ycomp = FinalData[twoData,1].flatten()
zcomp = FinalData[twoData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'g.')
# plot fours
xcomp = FinalData[fourData,0].flatten()
ycomp = FinalData[fourData,1].flatten()
zcomp = FinalData[fourData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,0].flatten()
ycomp = FinalData[sevenData,1].flatten()
zcomp = FinalData[sevenData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[oneData,0].flatten()
ycomp = FinalData[oneData,1].flatten()
zcomp = FinalData[oneData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('1st, 2nd and 4th principal components')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('4th PC')
#set legend for different colour data point
rpatch = mpatches.Patch(color='r', label='0')
gpatch = mpatches.Patch(color='g', label='2')
mpatch = mpatches.Patch(color='m', label='4')
ypatch = mpatches.Patch(color='y', label='7')
bpatch = mpatches.Patch(color='b', label='1')
ax.legend(handles=[rpatch, gpatch, mpatch,ypatch,bpatch], fontsize = 'large',loc = 'upper left')
plt.show()


# figure #second third and fourth PC
fig = plt.figure()
ax = fig.gca(projection = '3d')
# plot zeros
xcomp = FinalData[zeroData,1].flatten()
ycomp = FinalData[zeroData,2].flatten()
zcomp = FinalData[zeroData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,1].flatten()
ycomp = FinalData[twoData,2].flatten()
zcomp = FinalData[twoData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'g.')
# plot fours
xcomp = FinalData[fourData,1].flatten()
ycomp = FinalData[fourData,2].flatten()
zcomp = FinalData[fourData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,1].flatten()
ycomp = FinalData[sevenData,2].flatten()
zcomp = FinalData[sevenData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[oneData,1].flatten()
ycomp = FinalData[oneData,2].flatten()
zcomp = FinalData[oneData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('2nd,3rd and 4th principal components')
ax.set_xlabel('2nd PC')
ax.set_ylabel('3rd PC')
ax.set_zlabel('4th PC')
#set legend for different colour data point             
rpatch = mpatches.Patch(color='r', label='0')
gpatch = mpatches.Patch(color='g', label='2')
mpatch = mpatches.Patch(color='m', label='4')
ypatch = mpatches.Patch(color='y', label='7')
bpatch = mpatches.Patch(color='b', label='1')
ax.legend(handles=[rpatch, gpatch, mpatch,ypatch,bpatch], fontsize = 'large',loc = 'upper left')
plt.show()


# In[ ]:

fig = plt.figure()
ax = fig.gca(projection = '3d')

#plot data without labels
xcomp = FinalData[:,0].flatten()
ycomp = FinalData[:,1].flatten()
zcomp = FinalData[:,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('1st, 2nd and 3rd principal components without labels')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')

plt.show()


# In[ ]:

# figure #first second and fourth PC(124)
fig = plt.figure()
ax = fig.gca(projection = '3d')

#plot data without labels
xcomp = FinalData[:,0].flatten()
ycomp = FinalData[:,1].flatten()
zcomp = FinalData[:,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('1st, 2nd and 3rd principal components without labels')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('4th PC')
plt.show()


# In[ ]:

# figure #second third and fourth PC(234)
fig = plt.figure()
ax = fig.gca(projection = '3d')

#plot data without labels
xcomp = FinalData[:,1].flatten()
ycomp = FinalData[:,2].flatten()
zcomp = FinalData[:,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
ax.set_title('2nd,3rd and 4th principal components without labels')
ax.set_xlabel('2nd PC')
ax.set_ylabel('3rd PC')
ax.set_zlabel('4th PC')
plt.show()


# In[ ]:



