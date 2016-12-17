
# coding: utf-8

# In[ ]:

import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
#%matplotlib inline 
plt.ion()

plt.close('all')
#Load data
train = np.genfromtxt ('digits/train.csv', delimiter=",")
trainlabels = np.genfromtxt ('digits/trainlabels.csv', delimiter=",")

[n,m]  = np.shape(train)                    # number of pixels and number of training data(n=784,m=5000)
eta    = 0.05                               # learning rate
winit  = 1                                  # parameter controlling magnitude of initial conditions

maxT   = 40000                              #max time for running
digits = 15                                 #how many groups are trained here. 
counter = np.zeros((1,digits))              # counter for the winner neurons
wCount = np.ones((1,maxT+1)) * 0.25         # running avg of the weight change over time
alpha = 0.999

#normalize the train data
normal = np.dot(train.T,train)
normal = np.sqrt(np.diag(normal))
train_new = train / np.matlib.repmat(normal,n,1)

W = winit * np.random.rand(digits,n)        # Initialize Weight matrix (rows = output neurons, cols = input neurons)
normW = np.sqrt(np.diag(W.dot(W.T)))        # 15 by 1
normW = normW.reshape(digits,-1)            # reshape normW into a numpy 2d array

W = W / np.matlib.repmat(normW.T,n,1).T     # normalise using repmat(digits by 784)
                                            # normalise using numpy broadcasting -  
                                            #http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html
yl = int(round(digits/5))                   # counter for the rows of the subplot
if digits % 5 != 0:
    yl += 1

    
fig_neurs, axes_neurs = plt.subplots(yl,5)  # fig for the output neurons
fig_stats, axes_stats = plt.subplots(5,1)   # fig for the learning stats
plt.show()
bias = 0                                    # initialize bias
for t in range(1,maxT):
    i = math.ceil(m * np.random.rand())-1   # get a randomly generated index in the input range
    x = train_new[:,i]                      # pick a training instance using the random index(one data point)(784 by 1)

    h = W.dot(x)/digits                     # get output firing
    h = h.reshape((h.shape[0],-1))          # reshape h into a numpy 2d array

    xi = np.random.rand(digits,1) / 200     # add noise
    output = np.max(h+xi)                   # get the max in the output firing vector + noise
    k = np.argmax(h+xi)                     # get the index of the firing neuron

    gamma = 0.001
    bias = gamma*(1/n - x[k])               # create a bias
    
    counter[0,k] += 1                       # increment counter for winner neuron(1 by 15, ndarray)

    dw = eta * (x.T - W[k,:])               # calculate the change in weights for the k-th output neuron
                                            # get closer to the input (x - W)

    wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)

    W[k,:] = W[k,:] + dw                    # weights for k-th output are updated(num of digits by 784)
    h[k,:] = W[k,:].dot(x)/digits + bias    # add bias to winner freq
    
    # draw plots for the first timestep and then every 300 iterations
    if not t % 300 or t == 1:
        for ii in range(yl):                #plot from first row to others
            for jj in range(5):
                if 5*ii+jj < digits:
                    output_neuron = W[5*ii+jj,:].reshape((28,28),order = 'F')
                    #axes_neurs[ii,jj].text( verticalalignment='bottom', horizontalalignment='right',trainlabels[i])
                    axes_neurs[ii,jj].clear()
                    axes_neurs[ii,jj].imshow(output_neuron, interpolation='nearest')
                axes_neurs[ii,jj].get_xaxis().set_ticks([])
                axes_neurs[ii,jj].get_yaxis().set_ticks([])
        plt.draw()
        plt.pause(0.0001)


        # plot stats
        axes_stats[0].clear()
        axes_stats[0].set_title("Output firing rate")
        axes_stats[0].bar(np.arange(1,digits+1),h,align='center')
        axes_stats[0].set_xticks(np.arange(1,digits+1))
        axes_stats[0].relim()
        axes_stats[0].autoscale_view(True,True,True)

        axes_stats[1].clear()
        axes_stats[1].set_title("input image")
        axes_stats[1].imshow(x.reshape((28,28), order = 'F'), interpolation = 'nearest')
        axes_stats[1].get_xaxis().set_ticks([])
        axes_stats[1].get_yaxis().set_ticks([])

        axes_stats[2].clear()
        axes_stats[2].set_title("prototype image")
        axes_stats[2].imshow(W[k,:].reshape((28,28), order = 'F'), interpolation = 'nearest')
        axes_stats[2].get_xaxis().set_ticks([])
        axes_stats[2].get_yaxis().set_ticks([])

        axes_stats[3].clear()
        axes_stats[3].set_title("weights change over time")
        axes_stats[3].plot(wCount[0,2:t+1],'-b', linewidth=2.0)
        axes_stats[3].set_ylim([-0.001, 0.255])

        axes_stats[4].clear()
        axes_stats[4].set_title(" of neurons firing")
        axes_stats[4].bar(np.arange(1,digits+1),counter.T,align='center')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)

# click anywhere on the stats plot to close both figures
plt.waitforbuttonpress()

# calculate the counter to find dead units
counter = np.array(counter)
#print(counter)
n = 0
for i in range(counter.size):
    if counter[0, i] < 31:    
        n = n + 1
print(n)


plt.show()
## plot the figure of average change of weight
for t in range(1,maxT):
 
    axesW = plt.subplot(1,1,1)

    if not t % 300 or t == 1:
        
        plt.draw()
        axesW.clear()
        axesW.set_title("Average weight change over time")
        axesW.semilogx(wCount[0, 2:t + 1], '-r', linewidth=2.0)     # range from 2~t+1 (t-2 = 40000-2 = 39998)
                                                                    # semi-log x axis
        axesW.set_xlabel("time") 
        axesW.set_ylabel("change")
        axesW.set_ylim([0, 0.26])                                   # initial w is 0.249 and final w is 0.25

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)

# correlation matrix
corrW = np.corrcoef(W)
plt.matshow(corrW)
plt.xlabel("Prototypes 1-15")
plt.ylabel("Prototypes 1-15")
plt.title("The correlation matrix of prototypes")
plt.colorbar()

