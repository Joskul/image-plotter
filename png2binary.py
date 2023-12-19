"""
PNG to binary converter
"""
from PIL import Image

def make_non_transparent_pixels_black(input_path, output_path):
    # Open the image
    image = Image.open(input_path)

    # Ensure the image has an alpha channel (transparency)
    image = image.convert("RGBA")

    # Get the image data
    data = image.getdata()

    # Create a new list of pixel values with non-transparent pixels set to black
    new_data = [(255, 255, 255, 255) if alpha >= 128 else (0, 0, 0, 255) for (r, g, b, alpha) in data]

    # Update the image with the new pixel values
    image.putdata(new_data)

    # Save the modified image
    image.save(output_path, "PNG")

if __name__ == "__main__":
    # Replace 'input_image.png' with the path to your input PNG image
    input_path = 'images/input.PNG'
    
    # Replace 'output_image.png' with the desired output path
    output_path = 'images/bottle1.PNG'

    make_non_transparent_pixels_black(input_path, output_path)
