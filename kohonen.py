"""Python script for Exercise set 6 of the Unsupervised and
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import os.path as path
import pickle as pkl

def load_data(name='Bertrand Champenois'):
    """ Loads the required digits for the given name. Also serialises them to
    disk for faster future loading
    Returns: (data, labels, target_digits) -- data is only the subset of 4 selected digits
    """
    target_digits = name2digits(name) # assign the four digits that should be used
    print('Your digits: ', target_digits)

    # labels is small and fast, no need to serialise
    labels = np.loadtxt('labels.txt')
    target_idxs = np.logical_or.reduce([labels==x for x in target_digits])
    labels = labels[target_idxs]   # filter our digits only


    # data is big and takes time; check serialised version of subset
    target_digits_path = name + '.pkl'
    if path.isfile(target_digits_path):
        print('Loading from binary...')
        with open(target_digits_path, 'rb') as digits_file:
            data = pkl.load(digits_file)
    else:
        print('Loading from text...')
        data = np.loadtxt('data.txt')
        data = data[target_idxs,:]  # filter our digits only

        # serialise to disk:
        with open(target_digits_path, 'wb') as digits_file:
            pkl.dump(data, digits_file)
            print('Target digits serialised to %s' % target_digits_path)

    return data, labels, target_digits

def standardize(data, mean=None, std=None):
    """ Standardizes the data to have 0 mean and unit variance for each feature.
        If not given, these values are calculated from the data.
        Otherwise, then they are applied directly (as for testing data)

        Returns: data_transformed, mean, std
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    data = data - mean
    if std is None:
        std = np.std(data, axis=0)
    data[:, std > 0] = data[:, std > 0] / std[std > 0]  # avoid / 0

    return data, mean, std

def de_standardize(data, mean, std):
    """ Takes standardized data and shifts it by mean with std"""
    return data * std + mean

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')
    data, labels = load_data()  # default name

    # Kohonen algorithm hyper-parameters
    size_k = 6       # size of the Kohonen map (size_k, size_k)
    sigma  = 1       # width of the gaussian neighborhood
    eta    = 0.1     # learning rate
    tmax   = 5*2000  # max iteration count; substitutes convergence criterion

    # initialise the centers randomly
    dim = data.shape[1]     # 28*28 = 784
    data_range = 255.0
    centers = np.random.rand(size_k**2, dim) * data_range

    # build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    # set the random order in which the datapoints should be presented
    idxs_random = np.arange(tmax) % data.shape[0]
    np.random.shuffle(idxs_random)

    movs = [] # movements created at each step

    for example_idx in idxs_random:
        mov, win = som_step(centers, data[example_idx,:],neighbor,eta,sigma)
        movs.append(mov)
    plb.plot(movs)
    plb.show()

    for j in range(size_k ** 2):
        plb.subplot(size_k, size_k, j + 1)
        plb.imshow(np.reshape(centers[j, :], [28, 28]), interpolation='nearest', cmap='Greys')
        plb.axis('off')
    plb.show()


def som_step(centers,data,neighbor,eta,sigma):
    """ Performs one step of the sequential learning for a self-organized map (SOM).

        centers = som_step(centers,data,neighbor,eta,sigma)

    Input and output arguments:
        centers  (matrix) cluster centres. Have to be in format:
                          center X dimension
        data     (vector) the actual datapoint to be presented in this timestep
        neighbor (matrix) the layout of the centers in the desired neighborhood
        eta      (scalar) a learning rate
        sigma    (scalar) the width of the gaussian neighborhood function.
                          Effectively describing the width of the neighborhood

    Return:      (tuple)
        movement          total movement created by this `data` normalised by map size
        winner            the center closest to this data point
    """

    size_k = int(np.sqrt(len(centers)))

    # find the best matching unit via the minimal distance to the datapoint
    winner = np.argmin(np.sum((centers - data)**2, axis=1))
    win_x, win_y = np.nonzero(neighbor == winner)

    # total movement produced by this example
    movement = 0

    # update all units
    for c in range(size_k**2):
        # find coordinates of this unit
        c_x, c_y = np.nonzero(neighbor==c)

        # calculate the distance and discounting factor
        dist = np.sqrt((win_x-c_x)**2 + (win_y-c_y)**2)
        disc = gauss(dist, 0,sigma)

        # update weights and accumulate movement
        c_movement    = disc * (data - centers[c,:])
        centers[c,:] += eta * c_movement
        movement     += np.linalg.norm(c_movement)

    return movement / (size_k**2), winner


def gauss(x,m,s):
    """Return the gauss function N(x), with mean m and std s.
    Normalized such that N(x=m) = 1.
    """
    return np.exp((-(x - m)**2) / (2 * s*s))

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
    t = int(np.mod(s,x.shape[0]))

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

