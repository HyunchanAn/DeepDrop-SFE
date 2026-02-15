
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_droplet(ax, contact_angle, title):
    theta = np.radians(contact_angle)
    
    # Baseline
    ax.plot([-1.5, 1.5], [0, 0], color='black', linewidth=2)
    
    # Droplet profile (Spherical cap approximation)
    if contact_angle < 90:
        # Hydrophilic
        R = 1 / np.sin(theta)
        y_center = -R * np.cos(theta)
        
        # Draw arc
        phi = np.linspace(np.pi/2 - theta + 0.1, np.pi/2 + theta - 0.1, 100) # trim slightly for visual
        # Better parameterization
        # Circle equation: x^2 + (y - y_c)^2 = R^2
        # Intersection at y=0 is x = +/- 1 (normalized base radius = 1)
        # R = 1/sin(theta)
        # y_c = -1/tan(theta) 
        
        base_radius = 1.0
        R = base_radius / np.sin(theta)
        y_c = -base_radius / np.tan(theta)
        
        # Arc top
        t = np.linspace(np.radians(270 + contact_angle), np.radians(90 - contact_angle), 100)
        # Using simple wedge for fill
        
        circle = plt.Circle((0, y_c), R, color='skyblue', alpha=0.6, transform=ax.transData)
        ax.add_patch(circle)
        
        # Clip below y=0
        ax.set_ylim(0, 2.0)
        
    else:
        # Hydrophobic
        base_radius = 1.0
        # For > 90, center is above y=0
        # R = 1 / sin(180 - theta)
        R = base_radius / np.sin(np.pi - theta)
        y_c = -base_radius / np.tan(theta) # This becomes positive
        
        circle = plt.Circle((0, y_c), R, color='skyblue', alpha=0.6, transform=ax.transData)
        ax.add_patch(circle)
        ax.set_ylim(0, 2.5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{title}\n({contact_angle}°)", fontsize=10, y=-0.2)

# Setup plot
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
angles = [10, 45, 90, 120, 150]
labels = ["Super-hydrophilic", "Hydrophilic", "Neutral", "Hydrophobic", "Super-hydrophobic"]

for ax, angle, label in zip(axes, angles, labels):
    draw_droplet(ax, angle, label)

plt.tight_layout()
os.makedirs("demo/assets", exist_ok=True)
plt.savefig("demo/assets/contact_angle_ref.png", dpi=300, bbox_inches='tight')
print("Reference chart generated at demo/assets/contact_angle_ref.png")
