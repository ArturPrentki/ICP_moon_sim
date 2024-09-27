import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Constants
n_points = 50  # Number of landmarks
surface_function = lambda x, y: (x**2 + y)

landmarks_2D = np.random.rand(2, n_points) * 10 - 5  # x, y coordinates
landmarks_3D = np.vstack((landmarks_2D, surface_function(landmarks_2D[0], (landmarks_2D[1] + 50))))  # z values

def extract_theta_from_rotation_matrix(R_est):
    theta = np.arctan2(R_est[1, 0], R_est[0, 0])
    return theta

def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def sigma_points(mu, sigma, kappa):
    n = len(mu)
    sigma_points = np.zeros((2 * n + 1, n))
    lambda_ = kappa - n
    sqrt_sigma = np.linalg.cholesky((lambda_ + n) * sigma)

    sigma_points[0] = mu
    for i in range(n):
        sigma_points[i + 1] = mu + sqrt_sigma[:, i]
        sigma_points[n + i + 1] = mu - sqrt_sigma[:, i]

    return sigma_points

def unscented_transform(sigma_points, weights_mean, weights_cov):
    mu = np.sum(weights_mean[:, None] * sigma_points, axis=0)
    diff = sigma_points - mu
    sigma = np.einsum('i,ij,ik->jk', weights_cov, diff, diff)

    return mu, sigma

def transition_model(mu, u):
    mu_prime = mu.copy()
    mu_prime[0] += u[0] * np.cos(mu[2])
    mu_prime[1] += u[0] * np.sin(mu[2])
    mu_prime[2] += u[2]
    mu_prime[2] = normalize_angle(mu_prime[2])
    mu_prime[2] = surface_function(mu_prime[0], mu_prime[1])
    return mu_prime

def measurement_model(mu):
    return mu[:2] 

def prediction(mu, sigma, u, kappa):
    n = len(mu)
    sigma_points_ = sigma_points(mu, sigma, kappa)

    lambda_ = kappa - n
    alpha = 1e-3
    beta = 2.0
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = weights_mean[0] + (1 - alpha ** 2 + beta)

    propagated_sigma_points = np.array([transition_model(sp, u) for sp in sigma_points_])
    mu_pred, sigma_pred = unscented_transform(propagated_sigma_points, weights_mean, weights_cov)

    return mu_pred, sigma_pred

def update(mu, sigma, z, kappa):
    n = len(mu)
    sigma_points_ = sigma_points(mu, sigma, kappa)

    # Measurement prediction
    predicted_measurements = np.array([measurement_model(sp) for sp in sigma_points_])

    # Weights
    lambda_ = kappa - n
    alpha = 1e-3
    beta = 2.0
    weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = weights_mean[0] + (1 - alpha ** 2 + beta)

    z_pred, sigma_z = unscented_transform(predicted_measurements, weights_mean, weights_cov)

    # Measurement residual
    innovation = z - z_pred
    innovation[1] = normalize_angle(innovation[1])  

    # Calculate the cross-covariance
    cross_cov = np.zeros((n, 2))
    for i in range(len(weights_mean)):
        diff_x = sigma_points_[i] - mu
        diff_z = predicted_measurements[i] - z_pred
        cross_cov += weights_cov[i] * np.outer(diff_x, diff_z)

    # Kalman gain
    S = sigma_z + np.eye(2) * 0.1 
    K = cross_cov @ np.linalg.inv(S)

    # Update state
    mu += K @ innovation
    mu[2] = normalize_angle(mu[2])  
    sigma -= K @ S @ K.T

    return mu, sigma

def plot_state(mu, sigma, landmarks):
    plt.clf()
    plt.plot(mu[0], mu[1], 'bo', label='UKF Estimate')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='x', label='Landmarks')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('UKF State and Covariance')
    plt.legend()
    plt.grid()
    plt.pause(0.1)

def main():
    num_steps = 100
    ground_truth = np.zeros((num_steps, 3))
    ukf_estimates = np.zeros((num_steps, 3))
    ground_truth[0] = np.array([0.0, -0.5, surface_function(0.0, -0.5)])

    # Initial state
    mu = np.array([0.0, -0.5, surface_function(0.0, -0.5)])
    sigma = np.eye(3)

    ukf_estimates[0] = mu

    # Simulated control inputs
    u = np.array([0.1, 0.1, 0.05])
    kappa = 1.0
    landmarks = landmarks_3D.T

    for t in range(1, num_steps):
        # Simulate the robot's motion
        ground_truth[t] = transition_model(ground_truth[t - 1], u)

        # UKF Prediction step
        mu, sigma = prediction(mu, sigma, u, kappa)

        # Simulate a measurement
        z = measurement_model(ground_truth[t])

        # UKF Update step
        mu, sigma = update(mu, sigma, z, kappa)

        ukf_estimates[t] = mu

        if t % 5 == 0:
            plot_state(mu, sigma, landmarks[:, :2])

    # Visualization of ground truth vs estimated positions
    plt.figure()
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', color='g', linewidth=2)
    plt.plot(ukf_estimates[:, 0], ukf_estimates[:, 1], label='UKF Estimates', color='b', linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Robot Position Estimation vs Ground Truth (UKF)')
    plt.legend()
    plt.grid()
    plt.show()

   # Visualization of ground truth vs estimated positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid points for the surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = surface_function(X, Y)

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='cyan', edgecolor='none')

    # Plot ground truth trajectory
    ground_truth_z = surface_function(ground_truth[:, 0], ground_truth[:, 1])  # Calculate z-values from the surface
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth_z, label='Ground Truth', color='g', linewidth=2)
    
    # Plot UKF estimates
    ukf_estimates_z = surface_function(ukf_estimates[:, 0], ukf_estimates[:, 1])  # Calculate z-values from the surface
    ax.plot(ukf_estimates[:, 0], ukf_estimates[:, 1], ukf_estimates_z, label='UKF Estimates', color='b', linestyle='--')
    
    # Plot landmarks
    ax.scatter(landmarks_3D[0, :], landmarks_3D[1, :], landmarks_3D[2, :], color='r', marker='x', label='Landmarks')

    # Formatting the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Robot Position Estimation vs Ground Truth')
    ax.legend()
    ax.grid()
    plt.show()
if __name__ == '__main__':
    main()
