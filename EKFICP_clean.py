import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import linalg

n_points = 50
P_world = np.random.rand(3, n_points) * 10 - 5
# P_world[2, :] = np.abs(P_world[2, :])
# P_world = P_world / np.linalg.norm(P_world, axis=0)

def extract_theta_from_rotation_matrix(R_est):

    theta = np.arctan2(R_est[1, 0], R_est[0, 0])
    return theta

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
    e = R_est @ n - n_prime
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
            e, J = error_and_jacobian(R_est, P[:, i], Z[:,i])
            
            chi = (e.T @ e)
            if chi > kernel_threshold:
                e *= np.sqrt(kernel_threshold / chi)
                chi = kernel_threshold
            else:
                num_inliers[iteration] += 1
            chi_stats[iteration] += chi
            H += (J.T @ J)
            b += (J.T @ e)
        # print("e: ", e)
        # eigenvalues, eigenvectors = linalg.eig(H)
        # print("eigenvalues: ", eigenvalues)
        H += np.eye(3) * damping
        delta_alpha = -np.linalg.solve(H, b)
        
        # Update the estimate
        R_delta = R.from_rotvec(delta_alpha).as_matrix()
        R_est = R_delta @ R_est
        transformations.append(R_est)
        theta_est = extract_theta_from_rotation_matrix(R_est)
    return R_est, theta_est, chi_stats, num_inliers, transformations

def visualize_directions(P, Z, R_true, R_guess, R_result, transformations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Original directions
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, P[0, i], P[1, i], P[2, i], color='r', arrow_length_ratio=0.1, label='Original Directions' if i == 0 else "")
    
    # True transformed directions
    Z_true = R_true @ P
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_true[0, i], Z_true[1, i], Z_true[2, i], color='g',arrow_length_ratio=0.1, label='True Transformed Directions' if i == 0 else "")
    
    # Estimated transformed directions
    Z_result = R_result @ P
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_result[0, i], Z_result[1, i], Z_result[2, i], color='y',arrow_length_ratio=0.1, label='Estimated Transformed Directions' if i == 0 else "")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(P.shape[1]):
        ax.quiver(0, 0, 0, Z_true[0, i], Z_true[1, i], Z_true[2, i], color='g', label='True Transformed Directions' if i == 0 else "")
        ax.quiver(0, 0, 0, Z_result[0, i], Z_result[1, i], Z_result[2, i], color='y', label='Estimated Transformed Directions' if i == 0 else "")
    
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

def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def plot_state(mu, sigma, landmarks, observations):
    plt.clf()
    
    # Plot the EKF estimate as a point
    plt.plot(mu[0], mu[1], 'bo', label='EKF Estimate')
    
    # Plot the landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='x', label='Landmarks')
    
    # Plot the observations (bearings)
    if observations is not None:
        for obs in observations:
            lx = landmarks[obs['id'], 0]
            ly = landmarks[obs['id'], 1]
            plt.plot([mu[0], lx], [mu[1], ly], 'g-', alpha=0.5)  # Draw a line to the landmark
    
    # Draw covariance ellipse
    if sigma is not None:
        cov_ellipse(mu, sigma[:2, :2], edgecolor='blue', facecolor='blue')
    
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('EKF State and Covariance')
    plt.legend()
    plt.grid()
    plt.pause(0.1)

def cov_ellipse(mu, sigma, nstd=2, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(sigma[:2, :2])
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellip = Ellipse(xy=(mu[0], mu[1]), width=width, height=height, angle=angle, **kwargs)
    ellip.set_alpha(0.5)
    plt.gca().add_patch(ellip)

def transition_model(mu, u):
    mu_prime = mu.copy()
    mu_prime[0] += u[0] * np.cos(mu[2])
    mu_prime[1] += u[0] * np.sin(mu[2])
    mu_prime[2] += u[2]
    mu_prime[2] = normalize_angle(mu_prime[2])
    return mu_prime

def prediction(mu, sigma, transition):
    u = transition
    mu = transition_model(mu, u)
    
    Fx = np.array([[1, 0, -u[0] * np.sin(mu[2])],
                   [0, 1,  u[0] * np.cos(mu[2])],
                   [0, 0,  1]])
    
    Fi = np.eye(3)
    
    noise = 0.1
    motion_noise = np.diag([noise, noise, noise])
    
    sigma = Fx @ sigma @ Fx.T + Fi @ motion_noise @ Fi.T
    # sigma = Fx @ sigma @ Fx.T + Fi @ Fi.T
    
    return mu, sigma

def angleToRot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    num_steps = 100
    ground_truth = np.zeros((num_steps, 3))
    ekf_estimates = np.zeros((num_steps, 3))
    ground_truth[0] = np.array([0.0, -0.5, 0.0])
    
    # Initial state
    mu = np.array([0.0,-0.5, 0.0])
    sigma = np.eye(3)
    
    ekf_estimates[0] = mu
    
    # Simulated control inputs
    u = np.array([0.1, 0.1, 0.05])
    u_noise = u + np.random.normal(0, 0.01, u.shape)
    # Simulated landmark observations
    
    num_landmarks = n_points
    x_true = np.array([0, 0, 0, np.pi/2, np.pi/6, np.pi])
    R_true = R.from_rotvec([np.pi/4, -np.pi/6, np.pi/2]).as_matrix()

    iterations = 100
    damping = 0

    chi_stats = np.zeros((2, iterations))
    inliers_stats = np.zeros((2, iterations))

    R_guess = R.from_rotvec([0, 0, 0]).as_matrix()
    landmarks = P_world[:2, :].T
    for t in range(1, num_steps):
        # Simulate the robot's motion
        ground_truth[t] = transition_model(ground_truth[t-1], u)

        # EKF Prediction step
        mu, sigma = prediction(mu, sigma, u)
        R_mat=angleToRot(mu[2])
        
        tran=np.zeros((3,))
        # tran[:2]=mu[:2]
        P_robot = np.zeros((3, n_points))
        tran[0:1] = mu[0:1]
        tran[2] = 0
        gt_trans = np.zeros((3,))
        gt_trans[0:1] = ground_truth[t, 0:1]
        gt_trans[2] = 0
        Z=np.zeros((3,n_points))
        for i in range(n_points):
            P_robot[:, i] = P_world[:, i] 
            Z[:,i] = R_mat.T @ P_world[:, i] - tran

        dir_robot = P_robot/ np.linalg.norm(P_robot, axis=0)
        Z = Z / np.linalg.norm(Z, axis=0)
        # Perform ICP
        R_result, theta_icp, chi_stats, inliers_stats, transformations = do_icp(R_mat.T, Z ,dir_robot, iterations, damping, 1)
        print("theta_icp: ", theta_icp)
        print("ground_truth: ", ground_truth[t, 2])
        # EKF Correction step
        observations = []
        for i in range(num_landmarks):
            if np.random.rand() < 1:
                landmark_in_robot = angleToRot(ground_truth[t, 2])[0:2,0:2].T @ landmarks[i]
                dx = landmark_in_robot[0] - ground_truth[t, 0]
                dy = landmark_in_robot[1] - ground_truth[t, 1]
                observations.append({'id': i, 'x': dx, 'y': dy})
        ekf_estimates[t] = mu

        if t % 5 == 0:
            plot_state(mu, sigma, landmarks, observations)


if __name__ == '__main__':
    main()

