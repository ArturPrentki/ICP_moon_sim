import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def v2t(v):
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', v[3:6], degrees=False).as_matrix()
    T[:3, 3] = v[:3]
    return T

def error_and_jacobian(R_est, n, n_prime):
    # Error: e = R_est * n - n_prime
    e = R_est @ n - n_prime
    
    # Jacobian: J = [R_est * n]_x
    J = -skew_symmetric(R_est @ n)
    
    return e, J

def do_icp(R_guess, P, Z, num_iterations, damping, kernel_threshold):
    R_est = R_guess
    chi_stats = np.zeros(num_iterations)
    num_inliers = np.zeros(num_iterations)
    transformations = []
    
    for iteration in range(num_iterations):
        H = np.zeros((3, 3))
        b = np.zeros(3,)
        chi_stats[iteration] = 0
        
        for i in range(P.shape[1]):
            e, J = error_and_jacobian(R_est, P[:, i], Z[:, i])
            
            chi = (e.T @ e)
            if chi > kernel_threshold:
                e *= np.sqrt(kernel_threshold / chi)
                chi = kernel_threshold
            else:
                num_inliers[iteration] += 1
            chi_stats[iteration] += chi
            H += (J.T @ J)
            b += (J.T @ e)
        
        H += np.eye(3) * damping
        delta_alpha = -np.linalg.solve(H, b)
        
        # Update the estimate
        R_delta = R.from_rotvec(delta_alpha).as_matrix()
        R_est = R_delta @ R_est
        transformations.append(R_est)
    
    return R_est, chi_stats, num_inliers, transformations

def visualize_directions(P, Z, R_true, R_guess, R_result, transformations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Original directions
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, P[0, i], P[1, i], P[2, i], color='r', label='Original Directions' if i == 0 else "")
    
    # True transformed directions
    Z_true = R_true @ P
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_true[0, i], Z_true[1, i], Z_true[2, i], color='g', label='True Transformed Directions' if i == 0 else "")
    
    # Guessed transformed directions
    # Z_guess = R_guess @ P
    # for i in range(P.shape[1]):
    #     ax.quiver(0, 0, 0, Z_guess[0, i], Z_guess[1, i], Z_guess[2, i], color='b', label='Guessed Transformed Directions' if i == 0 else "")
    
    # Estimated transformed directions
    Z_result = R_result @ P
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_result[0, i], Z_result[1, i], Z_result[2, i], color='y', label='Estimated Transformed Directions' if i == 0 else "")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    # Plot the differences between true directions and estimated directions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_true[0, i], Z_true[1, i], Z_true[2, i], color='g', label='True Transformed Directions' if i == 0 else "")
        ax.quiver(0, 0, 0, Z_result[0, i], Z_result[1, i], Z_result[2, i], color='y', label='Estimated Transformed Directions' if i == 0 else "")
    
    # Add vectors showing the difference
    for i in range(P.shape[1]):
        ax.quiver(Z_true[0, i], Z_true[1, i], Z_true[2, i], 
                  Z_result[0, i] - Z_true[0, i], 
                  Z_result[1, i] - Z_true[1, i], 
                  Z_result[2, i] - Z_true[2, i], 
                  color='r', arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Differences between True and Estimated Directions')
    plt.show()

# Test code
n_points = 5
P_world = np.random.rand(3, n_points) * 10 - 5
P_world = P_world / np.linalg.norm(P_world, axis=0)  # Normalize to unit vectors

x_true = np.array([0, 0, 0, np.pi/2, np.pi/6, np.pi])
R_true = R.from_rotvec([np.pi/2, np.pi/6, np.pi]).as_matrix()
Z = R_true @ P_world

iterations = 100
damping = 0.6

chi_stats = np.zeros((2, iterations))
inliers_stats = np.zeros((2, iterations))

R_guess = R.from_rotvec([0.5, 0.5, 0.5]).as_matrix()

R_result, chi_stats[0, :], inliers_stats[0, :], transformations = do_icp(R_guess, P_world, Z, iterations, damping, 1)

visualize_directions(P_world, Z, R_true, R_guess, R_result, transformations)
