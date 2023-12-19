import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import re
from scipy.integrate import quad

def surface_area(poly_function, lower_bound, upper_bound):
    def derivative_squared(x):
        return (np.polyder(poly_function, 1)(x))**2 + 1

    result, _ = quad(lambda x: 2 * np.pi * np.sqrt(derivative_squared(x)), lower_bound, upper_bound) + (np.pi * polyfunction(0) ** 2) + (np.pi * polyfunction(upper_bound) ** 2)
    return result

def volume(poly_function, lower_bound, upper_bound):
    def disc_function(x):
        return 2 * np.pi * (np.abs(poly_function(x)) ** 2)

    result, _ = quad(disc_function, lower_bound, upper_bound)
    return result

def preprocess_image(image_path):
    image = io.imread(image_path)[:, :, :3]
    grayscale_image = color.rgb2gray(image)
    threshold_value = threshold_otsu(grayscale_image)
    binary_image = grayscale_image > threshold_value
    return binary_image

def extract_upper_contour(binary_image, scale_factor=1.0):
    upper_contour = np.argmax(binary_image, axis=0)
    scaled_upper_contour = np.column_stack([upper_contour, np.arange(len(upper_contour))]) * scale_factor
    return scaled_upper_contour

def fit_polynomial(upper_contour, degree=5):
    # Adjust the degree and coefficients as needed
    coefficients = np.polyfit(upper_contour[:, 1], upper_contour[:, 0], degree)

    # Negate the coefficients to make the function negative
    negated_coefficients = -coefficients

    # Create the negative polynomial
    negative_polynomial = np.poly1d(negated_coefficients)
    
    return negative_polynomial

def convert_to_scientific_notation(input_str):
    matches = re.findall(r"\(([-+]?\d+(\.\d+)?)[eE]([-+]?\d+)\)", input_str)

    for match in matches:
        full_match = match[0]
        coefficient = match[0]
        exponent = match[2]

        if coefficient == '1':
            replacement = f"10^{{{exponent}}}"
        else:
            replacement = f"{coefficient}\cdot10^{{{exponent}}}"

        # Create the scientific notation string and replace in the input string
        input_str = input_str.replace(f"({full_match}e{exponent})", replacement)

    return input_str
    
def desmos_equation(poly_function, binary_image, scale_factor):
    terms = [f"({coeff:.3e})x^{{{degree}}}" for degree, coeff in enumerate(reversed(poly_function.coefficients))]
    equation = " + ".join(terms)
    print(poly_function)
    
    desmos_eq = f"{binary_image.shape[0] * scale_factor} + {convert_to_scientific_notation(equation)}{{0<=x<={binary_image.shape[1] * scale_factor}}}".replace(" + -", " - ").replace("-0","-")

    return desmos_eq

def plot_polynomial(poly_function, upper_contour, binary_image, scale_factor):
    # Generate x values
    x_vals = np.linspace(0, max(upper_contour[:, 1]), 1000)

    # Calculate corresponding y values using the negative polynomial
    y_vals = -poly_function(x_vals)

    # Scatter plot with non-inverted y-axis
    plt.scatter(upper_contour[:, 1], upper_contour[:, 0], color='red', label='Upper Contour')

    # Display the binary image
    plt.imshow(binary_image, extent=[0, binary_image.shape[1] * scale_factor, 0, binary_image.shape[0] * scale_factor], cmap='gray', alpha=0.2, origin='lower')  # Adjust 'extent' and 'origin'

    # Plot the inverted polynomial
    plt.plot(x_vals, y_vals, label='Fitted Polynomial')

    plt.xlabel('X (Physical Units)')
    plt.ylabel('Y (Physical Units)')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.legend()
    plt.show()


def main():
    # Specify the path to the input image
    image_path = 'images/bottle2.png'
    physical_height = 23.5
    degree = 6

    binary_image = preprocess_image(image_path)

    scale_factor = physical_height /binary_image.shape[1]

    upper_contour = extract_upper_contour(binary_image, scale_factor=scale_factor)
    poly_function = fit_polynomial(upper_contour, degree)

    equation = desmos_equation(poly_function, binary_image, scale_factor)
    print("Desmos-compatible equation:")
    print(equation)

    plot_polynomial(poly_function, upper_contour, binary_image, scale_factor)

    # Calculate surface area
    sa = surface_area(poly_function, 0, physical_height)
    print(f"Surface Area: {sa:.4f} square units")

    # Calculate volume
    vol = volume(poly_function, 0, physical_height)
    print(f"Volume: {vol:.4f} cubic units")

if __name__ == "__main__":
    main()
