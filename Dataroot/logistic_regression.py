import  numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def newton(features,predictions):
    m = features.shape[0]
    H = np.dot(np.dot((1 / m) * features.T, np.diag(predictions)), np.dot(np.diag(1 - predictions), features))
    return H

def log_likelihood(features, target, weights):
    '''
          U = sum(target * weights_tr * features - log(1 + exp(weights_tr * features)))      '''

    scores = np.dot(features, weights)
    return np.sum(target * scores - np.log(1 + np.exp(scores)))

def grad(features, target, predictions):
    m = features.shape[0]
    return np.dot(1/m* features.T, predictions - target)

def logistic_regression(features, target, num_steps):
    # initialize weights
    features = np.hstack((np.ones((features.shape[0], 1)), features))
    weights = np.zeros(features.shape[1])
    converg = np.zeros((num_steps, 1));
    m = features.shape[0]

    # iterative process
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        gradient = grad(features, target, predictions)
        H = newton(features,predictions)
        weights -= np.linalg.lstsq(H, gradient)[0]
        converg[step] = (1/m)*np.sum(-target*np.log(predictions)-(1-target)*np.log(1-predictions))

    return weights,converg

def run(showPlots=False):
    np.random.seed(12)
    num_observations = 5000

    #Generate data
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),np.ones(num_observations)))

    num_steps=7
    weights,converg = logistic_regression(simulated_separableish_features, simulated_labels,
                                  num_steps=num_steps)

    data_with_intercept = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                     simulated_separableish_features))
    final_scores = np.dot(data_with_intercept, weights)
    preds = np.round(sigmoid(final_scores))
    print(preds)
    if showPlots:
        #preds plot
        plt.figure(figsize=(12, 8))
        plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                    c=preds == simulated_labels - 1, alpha=.8, s=50)

        #convergence plot for newton method
        plt.figure(figsize=(12, 8))
        plt.plot(range(0, num_steps), converg, marker='o', linestyle='--', color='r')
        plt.show()
def main():
   run(True)

if __name__ == "__main__":
    main()