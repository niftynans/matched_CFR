import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sierpinski3d(iterations):
    # Initial tetrahedron vertices
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(0.75), 0],
        [0.5, np.sqrt(0.75)/3, np.sqrt(2/3)]
    ])
    
    points = vertices.copy()
    
    # Generate points for each iteration
    for _ in range(iterations):
        new_points = []
        for i in range(len(points)):
            # Get random vertex from the tetrahedron
            vertex = vertices[np.random.randint(0, 4)]
            # Calculate midpoint between current point and random vertex
            midpoint = (points[i] + vertex) / 2
            new_points.append(midpoint)
        points = np.array(new_points)
    
    return points

# Create the visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Generate points
points = sierpinski3d(iterations=10000)

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.6, s=1)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Sierpinski Triangle (Tetrahedron)')

# Adjust the viewing angle for better visualization
ax.view_init(elev=20, azim=45)

plt.show()