import numpy as np
def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    indices = np.random.permutation(points.shape[0])
    points_indx = indices[:k]
    training = points[points_indx, :]

    return training

def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:,np.newaxis])**2).sum(axis=2))
    return np.argmin(distances,axis=0)

def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def run(points,num_iterations,k):

    # Initialize centroids
    centroids = initialize_centroids(points, k)

    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)

    return centroids

def main():
    points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                        (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                        (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))
    num_iterations = 100
    k = 3
    centroids = run(points,num_iterations,k)

if __name__ == "__main__":
    main()
