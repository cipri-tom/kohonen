"""Python script for Exercise set 6 of the Unsupervised and
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib import animation

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')

    dim = 28*28
    data_range = 255.0

    # load in data and labels
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    # select 4 digits
    name = 'Bertrand Champenois' # REPLACE BY YOUR OWN NAME
    targetdigits = name2digits(name) # assign the four digits that should be used
    print(targetdigits) # output the digits that were selected
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]

    dy, dx = data.shape

    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 6

    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = 1

    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range

    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = 0.01 # HERE YOU HAVE TO SET YOUR OWN LEARNING RATE

    #set the maximal iteration count
    tmax = 10*2000 # this might or might not work; use your own convergence criterion

    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)

    accumuls = [[] for i in range(36)]
    accumuls2 = [[] for i in range(36)]

    for t, i in enumerate(i_random):
        aa,bb = som_step(centers, data[i,:],neighbor,eta,sigma)
        accumuls[bb] += [aa]
        accumuls2[bb] += [t]
    for j in range(size_k ** 2):
        plb.figure()
        plb.title(j)
        plb.plot(accumuls2[j],np.convolve(accumuls[j],[1]*50)[:-49])
        plb.show()

    for j in range(size_k ** 2):
        plb.subplot(size_k, size_k, j + 1)
        plb.imshow(np.reshape(centers[j, :], [28, 28]), interpolation='nearest', cmap='Greys')
        plb.axis('off')
    plb.show()






    # for visualization, you can use this:


    # leave the window open at the end of the loop


    #plb.draw()



def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a
    self-organized map (SOM).

      centers = som_step(centers,data,neighbor,eta,sigma)

      Input and output arguments:
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """

    size_k = int(np.sqrt(len(centers)))

    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))
    bb = b
    accumul = np.linalg.norm(eta * (data - centers[b, :]))
    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)

    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights
        x = disc * eta * (data - centers[j,:])

        centers[j,:] +=  x

    return accumul,bb


def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.

     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """

    name = name.lower()

    if len(name)>25:
        name = name[0:25]

    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

    n = len(name)

    s = 0.0

    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.mod(s,x.shape[0])

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

