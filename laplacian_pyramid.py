import vpi

# Load image
img = vpi.imread("./yubit.JPG")

# Create Laplacian pyramid
# Initialize empty list to store pyramid levels
pyramid_levels = []

# Start with original image
current = img

# Generate pyramid levels
for i in range(4):  # Create 4 levels of pyramid
    # Create downsampled version
    downsampled = vpi.rescale(current, 0.5, interp=vpi.Interp.LINEAR)
    
    # Create upsampled version of downsampled image
    upsampled = vpi.rescale(downsampled, 2.0, interp=vpi.Interp.LINEAR)
    
    # Compute and store difference (Laplacian)
    laplacian = current - upsampled
    pyramid_levels.append(laplacian)
    
    # Update current for next iteration
    current = downsampled

# Add the final downsampled image as the top of the pyramid
pyramid_levels.append(current)

# Now pyramid_levels contains the Laplacian pyramid
# pyramid_levels[0] is highest frequency, pyramid_levels[-1] is lowest frequency

# Display the Laplacian pyramid
for i, level in enumerate(pyramid_levels):
    vpi.imshow(level, title=f"Laplacian Level {i}")
    vpi.show()