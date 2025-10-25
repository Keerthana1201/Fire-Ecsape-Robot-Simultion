import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFilter

def create_risk_overlay(risk_map, alpha=0.3):
    """Create a semi-transparent overlay showing risk levels."""
    if risk_map is None or risk_map.size == 0:
        return None
    
    # Create a colormap: green (low) -> yellow -> red (high)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'risk', 
        [(0, 1, 0, 0.3),   # Green, transparent
         (1, 1, 0, 0.5),   # Yellow, more opaque
         (1, 0, 0, 0.7)]   # Red, most opaque
    )
    
    # Normalize risk values
    norm_risk = risk_map / (risk_map.max() + 1e-6)  # Avoid division by zero
    
    # Apply colormap
    rgba = cmap(norm_risk)
    
    # Create image
    height, width = risk_map.shape
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    pixels = overlay.load()
    
    # Set pixels based on risk level
    for y in range(height):
        for x in range(width):
            r, g, b, a = rgba[y, x]
            pixels[x, y] = (
                int(r * 255),
                int(g * 255),
                int(b * 255),
                int(a * 255 * alpha)
            )
    
    # Apply blur for better visualization
    return overlay.filter(ImageFilter.GaussianBlur(radius=1.5))

def draw_risk_legend(canvas, x, y, width=200, height=20):
    """Draw a legend for the risk visualization."""
    # Create gradient
    gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    
    for i in range(width):
        # Interpolate color
        ratio = i / width
        if ratio < 0.33:
            # Green to yellow
            r = int(255 * (ratio * 3))
            g = 255
            b = 0
            a = int(100 + 55 * (ratio * 3))
        elif ratio < 0.66:
            # Yellow to orange
            r = 255
            g = int(255 * (2 - ratio * 3))
            b = 0
            a = int(155 + 55 * ((ratio - 0.33) * 3))
        else:
            # Orange to red
            r = 255
            g = int(255 * (1 - (ratio - 0.66) * 3))
            b = 0
            a = 210
        
        # Draw line
        draw.line([(i, 0), (i, height-1)], fill=(r, g, b, a), width=1)
    
    # Add text
    draw.text((5, 2), "Low Risk", fill=(0, 0, 0, 200))
    draw.text((width - 60, 2), "High Risk", fill=(0, 0, 0, 200))
    
    # Convert to PhotoImage and add to canvas
    from PIL import ImageTk
    photo = ImageTk.PhotoImage(gradient)
    label = canvas.create_image(x, y, image=photo, anchor='nw')
    
    # Keep reference to prevent garbage collection
    canvas.risk_legend = photo
    
    return label
