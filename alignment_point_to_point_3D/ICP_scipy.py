import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelmin
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# def Rx(rot_x):
#     c = np.cos(rot_x)
#     s = np.sin(rot_x)
#     R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
#     return R

# def Ry(rot_y):
#     c = np.cos(rot_y)
#     s = np.sin(rot_y)
#     R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
#     return R

# def Rz(rot_z):
#     c = np.cos(rot_z)
#     s = np.sin(rot_z)
#     R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
#     return R

# def rx_prime(rot_x):
#     dc = -np.sin(rot_x)
#     ds = np.cos(rot_x)
#     R = np.array([[0, 0, 0], [0, dc, -ds], [0, ds, dc]])
#     return R

# def ry_prime(rot_y):
#     dc = -np.sin(rot_y)
#     ds = np.cos(rot_y)
#     R = np.array([[dc, 0, ds], [0, 0, 0], [-ds, 0, dc]])
#     return R

# def rz_prime(rot_z):
#     dc = -np.sin(rot_z)
#     ds = np.cos(rot_z)
#     R = np.array([[dc, -ds, 0], [ds, dc, 0], [0, 0, 0]])
#     return R
def v2t(v):
    T = np.eye(4)
    T[:3, :3] = R.from_euler('XYZ', v[3:6]).as_matrix()
    T[:3, 3] = v[:3]
    return T
# def errorAndJacobian(x, p, z):
#     R_xyz = R.from_euler('xyz', x[3:6])
#     t = x[:3]
#     z_hat = R_xyz.apply(p) + t
#     e = z_hat - z
#     J = np.zeros((3, 6))
#     J[:3, :3] = np.eye(3)
    
#     Rx_prime = R.from_euler('x', 1e-6).as_matrix() - np.eye(3)
#     Ry_prime = R.from_euler('y', 1e-6).as_matrix() - np.eye(3)
#     Rz_prime = R.from_euler('z', 1e-6).as_matrix() - np.eye(3)
    
#     J[:3, 3] = (R_xyz.as_matrix() @ (Rx_prime @ p))
#     J[:3, 4] = (R_xyz.as_matrix() @ (Ry_prime @ p))
#     J[:3, 5] = (R_xyz.as_matrix() @ (Rz_prime @ p))
    
#     return e, J
# x = np.array([1, 2, 3, 4, 5, 6])
# p = np.array([1, 2, 3])
# z = np.array([4, 5, 6])
# J = errorAndJacobian(x, p, z)
# print("Jacobian matrix:")
# print(J)
# def doICP(x_guess, P, Z, num_iterations, damping, kernel_threshold):
#     x = x_guess
#     chi_stats = np.zeros(num_iterations)
#     num_inliers = np.zeros(num_iterations)
#     print_first_jacobian = True
#     for iteration in range(num_iterations):
#         H = np.zeros((6, 6))
#         b = np.zeros((6,))
#         chi_stats[iteration] = 0
        
#         for i in range(P.shape[1]):
#             e, J = errorAndJacobian(x, P[:, i], Z[:, i])
#             if print_first_jacobian and i == 0:
#                 # print("Jacobian:")
#                 # print(J)
#                 print_first_jacobian = False
#             chi = np.dot(e.T, e)
#             if chi > kernel_threshold:
#                 e *= np.sqrt(kernel_threshold / chi)
#                 chi = kernel_threshold
#             else:
#                 num_inliers[iteration] += 1
#             chi_stats[iteration] += chi
#             H += np.dot(J.T, J)
#             b += np.dot(J.T, e)
        
#         H += np.eye(6) * damping
#         dx = -np.linalg.solve(H, b)
#         x += dx
    
#     return x, chi_stats, num_inliers
# def doICP(x_guess, P, Z, num_iterations, damping, kernel_threshold):
#     x = x_guess
#     chi_stats = np.zeros(num_iterations)
#     num_inliers = np.zeros(num_iterations)
#     rx = Rx(x[3])
#     ry = Ry(x[4])
#     rz = Rz(x[5])
#     for iteration in range(num_iterations):
#         H = np.zeros((6, 6))
#         b = np.zeros(6,)
#         chi_stats[iteration] = 0
        
#         for i in range(P.shape[1]):
#             R_xyz = R.from_euler('XYZ', x[3:6])
#             t = x[:3]
#             z_hat = R_xyz.apply(P[:, i]) + t
#             e = z_hat - Z[:, i]
#             J = np.zeros((3, 6))
#             J[:3, :3] = np.eye(3)
            
#             # Rx_prime = R.from_euler('X', 1e-6).as_matrix() - np.eye(3)
#             # Ry_prime = R.from_euler('Y', 1e-6).as_matrix() - np.eye(3)
#             # Rz_prime = R.from_euler('Z', 1e-6).as_matrix() - np.eye(3)
#             Rx_prime = rx_prime(x[3])
#             Ry_prime = ry_prime(x[4])
#             Rz_prime = rz_prime(x[5])
            
#             J[:3, 3] = (R_xyz.as_matrix() @ (Rx_prime @ P[:, i]))
#             J[:3, 4] = (R_xyz.as_matrix() @ (Ry_prime @ P[:, i]))
#             J[:3, 5] = (R_xyz.as_matrix() @ (Rz_prime @ P[:, i]))
#             # J[:3, 3] = (Rx_prime @ (ry @ (rz @  P[:, i])))
#             # J[:3, 4] = (rx @ (Ry_prime @ (rz @  P[:, i])))
#             # J[:3, 5] = (rx @ (ry @ (Rz_prime @  P[:, i])))
#             # if iteration == 0 and i == 0:

#                 # print("Jacobian:")
#                 # print(J)
#             chi = (e.T @ e)
#             if chi > kernel_threshold:
#                 e *= np.sqrt(kernel_threshold / chi)
#                 chi = kernel_threshold
#             else:
#                 num_inliers[iteration] += 1
#             chi_stats[iteration] += chi
#             H += (J.T @ J)
#             b += (J.T @ e)
        
#         H += np.eye(6) * damping
#         dx = -np.linalg.solve(H, b)
#         x += dx
    
#     return x, chi_stats, num_inliers

# # Test code
# n_points = 100
# P_world = np.random.rand(3, n_points) * 10 - 5

# x_true = np.array([0, 0, 0, np.pi/2, np.pi/6, np.pi])
# X_true = v2t(x_true)
# P_world_hom = np.ones((4, n_points))
# P_world_hom[:3, :] = P_world
# Z_hom = (X_true @ P_world_hom)
# Z = Z_hom[:3, :]

# noise_sigma = 1
# # Z[:3, :] += np.random.normal(0, noise_sigma, (3, n_points))

# iterations = 100
# damping = 100

# chi_stats = np.zeros((2, iterations))
# inliers_stats = np.zeros((2, iterations))

# x_guess = x_true + np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

# x_result, chi_stats[0, :], inliers_stats[0, :] = doICP(x_guess, P_world, Z, iterations, damping, 1e9)
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.title('Good guess plain implementation')
# plt.xlabel('Iteration')
# plt.ylabel('Error (less is better)')
# plt.plot(np.log(chi_stats[0, :] + 1), "-b", linewidth=3)
# plt.show()

# x_guess = np.zeros((6))
# # x_guess = np.array([20, 20, 20, np.pi, np.pi/2, np.pi/2])
# x_result, chi_stats[1, :], inliers_stats[1, :] = doICP(x_guess, P_world, Z, iterations, damping, 1e9)
# # print(argrelmin(x_result))  
# plt.figure(2)
# plt.title('Good and bad guess, plain implementation')
# plt.xlabel('Iteration')
# plt.ylabel('Error (less is better)')
# plt.plot(np.log(chi_stats[0, :] + 1), "-b", linewidth=3)
# plt.plot(np.log(chi_stats[1, :] + 1), "-r", linewidth=3)
# plt.show()
def ekf_update(x, P, z, R, Q):
    # Prediction
    x = x
    P = P + Q
    
    # Update
    H = np.eye(3, 6)
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = np.dot(np.eye(6) - np.dot(K, H), P)
    return x, P
def doICP(x_guess, P, Z, num_iterations, damping, kernel_threshold):
    x = x_guess
    chi_stats = np.zeros(num_iterations)
    num_inliers = np.zeros(num_iterations)
    
    for iteration in range(num_iterations):
        H = np.zeros((6, 6))
        b = np.zeros(6,)
        chi_stats[iteration] = 0
        
        for i in range(P.shape[1]):
            R_xyz = R.from_euler('XYZ', x[3:6])
            t = x[:3]
            z_hat = R_xyz.apply(P[:, i]) + t
            e = z_hat - Z[:, i]
            J = np.zeros((3, 6))
            J[:3, :3] = np.eye(3)
            
            # Compute the correct Jacobian using finite differences
            epsilon = 1e-6
            for j in range(3, 6):
                x_temp = np.copy(x)
                x_temp[j] += epsilon
                R_temp = R.from_euler('XYZ', x_temp[3:6])
                z_hat_temp = R_temp.apply(P[:, i]) + x_temp[:3]
                J[:, j] = (z_hat_temp - z_hat) / epsilon
            
            chi = (e.T @ e)
            if chi > kernel_threshold:
                e *= np.sqrt(kernel_threshold / chi)
                chi = kernel_threshold
            else:
                num_inliers[iteration] += 1
            chi_stats[iteration] += chi
            H += (J.T @ J)
            b += (J.T @ e)
        
        H += np.eye(6) * damping
        dx = -np.linalg.solve(H, b)
        x += dx
    
    return x, chi_stats, num_inliers

# Test code
n_points = 100
P_world = np.random.rand(3, n_points) * 10 - 5

x_true = np.array([0, 0, 0, np.pi/2, np.pi/6, np.pi])
X_true = v2t(x_true)
P_world_hom = np.ones((4, n_points))
P_world_hom[:3, :] = P_world
Z_hom = (X_true @ P_world_hom)
Z = Z_hom[:3, :]

noise_sigma = 1
# Z[:3, :] += np.random.normal(0, noise_sigma, (3, n_points))

iterations = 100
damping = 100

chi_stats = np.zeros((2, iterations))
inliers_stats = np.zeros((2, iterations))

x_guess = x_true + np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

x_result, chi_stats[0, :], inliers_stats[0, :] = doICP(x_guess, P_world, Z, iterations, damping, 1e9)
plt.figure(1)
plt.title('Good guess plain implementation with noise')
plt.xlabel('Iteration')
plt.ylabel('Error (less is better)')
plt.plot(np.log(chi_stats[0, :] + 1), "-b", linewidth=3)
plt.show()

x_guess = np.zeros((6))
x_result, chi_stats[1, :], inliers_stats[1, :] = doICP(x_guess, P_world, Z, iterations, damping, 1e9)
plt.figure(2)
plt.title('Good and bad guess, plain implementation with noise')
plt.xlabel('Iteration')
plt.ylabel('Error (less is better)')
plt.plot(np.log(chi_stats[0, :] + 1), "-b", linewidth=3)
plt.plot(np.log(chi_stats[1, :] + 1), "-r", linewidth=3)
plt.show()