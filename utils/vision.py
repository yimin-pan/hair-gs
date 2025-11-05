import cv2
import numpy as np


def estimate_orientation_field(
    image, kernel_size=31, sigma=2, lambda_=3, gamma=0.5, num_angles=180
):
    """
    Estimate the 2D orientation field and confidence of the given image, resolution is 1 degree.
    """

    def angdiff(angle1, angle2):
        return np.pi / 2 - np.abs(np.abs(angle1 - angle2) - np.pi / 2)

    # Convert image to grayscale if it's not already
    gray = image

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Generate Gabor kernels for each orientation (1-degree steps)
    # orientations = np.arange(0, 180, 1) * np.pi / 180
    orientations = np.linspace(0, np.pi, num_angles)
    kernels = [
        cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            theta,
            lambda_,
            gamma,
            0,
            ktype=cv2.CV_32F,
        )
        for theta in orientations
    ]

    # Convolve the image with each kernel
    responses = [np.abs(cv2.filter2D(gray, -1, kernel)) for kernel in kernels]

    # Find the orientation with the maximum response
    responses = np.stack(responses, axis=2)
    max_response = np.argmax(responses, axis=2)
    orientation_field = orientations[max_response]

    # Compute confidence normalized to [0, 1]
    orientation_field_repeated = np.repeat(
        orientation_field[:, :, np.newaxis], len(orientations), axis=2
    )
    orientations_mat = np.ones((height, width, len(orientations))) * orientations
    diff = angdiff(orientation_field_repeated, orientations_mat)
    diff = diff * diff * responses
    sum_of_responses = np.sum(responses, axis=2)
    variance = np.sum(diff, axis=2) / (sum_of_responses + 1e-7)
    has_variance = variance != 0
    confidence = np.ones(orientation_field.shape, dtype=np.float32)
    valid_confidence = 1 / (variance * variance)[has_variance]
    max_confidence = np.max(valid_confidence)
    valid_confidence = valid_confidence / max_confidence
    confidence[has_variance] = valid_confidence

    return orientation_field, confidence
