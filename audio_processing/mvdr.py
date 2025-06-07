import numpy as np

def mvdr_beamforming(signal, mic_array_positions, look_direction):
    covariance_matrix = np.cov(signal)
    steering_vector = np.exp(-2j * np.pi * mic_array_positions.dot(look_direction))

    weights = np.linalg.solve(covariance_matrix, steering_vector)
    weights /= steering_vector.conj().T @ weights

    return np.real(np.dot(weights.conj().T, signal))
