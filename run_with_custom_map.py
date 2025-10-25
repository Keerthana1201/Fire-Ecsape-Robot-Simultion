import sys
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def convert_ascii_map(ascii_map):
    """Convert ASCII map to numeric grid format."""
    # Define character mappings
    char_to_num = {
        '#': 1,  # Wall
        '.': 0,  # Empty
        'F': 2,  # Fire
        'S': 0,  # Start (treated as empty, we'll track position separately)
        'E': 0,  # Exit (treated as empty, we'll track positions separately)
        'T': 0,  # Treat as empty for pathfinding
        'C': 0   # Treat as empty for pathfinding
    }
    
    # Find start and exit positions
    start_pos = None
    exits = []
    
    # First pass to find dimensions
    height = len(ascii_map)
    width = max(len(line) for line in ascii_map) if height > 0 else 0
    
    # Create grid
    grid = np.zeros((height, width), dtype=int)
    
    # Fill grid and track special positions
    for i, line in enumerate(ascii_map):
        for j, char in enumerate(line):
            if char == 'S' and start_pos is None:
                start_pos = (i, j)
            elif char == 'E':
                exits.append((i, j))
            grid[i, j] = char_to_num.get(char, 0)  # Default to 0 (empty) for unknown chars
    
    # If no start position found, use top-left empty cell
    if start_pos is None:
        for i in range(height):
            for j in range(width):
                if grid[i, j] == 0:
                    start_pos = (i, j)
                    break
            if start_pos is not None:
                break
    
    # If no exits found, use bottom-right empty cell
    if not exits:
        for i in range(height-1, -1, -1):
            for j in range(width-1, -1, -1):
                if grid[i, j] == 0:
                    exits = [(i, j)]
                    break
            if exits:
                break
    
    return grid, start_pos, exits

def load_map(map_path):
    """Load map from ASCII file."""
    with open(map_path, 'r') as f:
        ascii_map = [line.strip() for line in f if line.strip()]
    
    return convert_ascii_map(ascii_map)

class SimpleSimulation:
    def __init__(self, root, map_path):
        self.root = root
        self.root.title("Fire Escape Robot Simulation")
        
        # Load the map
        self.grid_data, self.start_pos, self.exits = load_map(map_path)
        self.robot_pos = list(self.start_pos)
        
        # Initialize fire
        self.fire = FireSpread(prob=0.1)
        
        # Setup UI
        self.setup_ui()
        
        # Add some initial fire
        self.add_initial_fire(5)
        
        # Start simulation
        self.running = False
        self.step_count = 0
        self.update_display()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create canvas
        cell_size = 20
        canvas_width = self.grid_data.shape[1] * cell_size
        canvas_height = self.grid_data.shape[0] * cell_size
        
        self.canvas = tk.Canvas(
            self.root, 
            width=canvas_width, 
            height=canvas_height,
            bg='white'
        )
        self.canvas.pack(pady=10)
        
        # Add controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_btn = tk.Button(
            control_frame, 
            text="▶ Start", 
            command=self.toggle_simulation,
            width=10
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.step_btn = tk.Button(
            control_frame,
            text="⏯ Step",
            command=self.step_simulation,
            width=10
        )
        self.step_btn.pack(side='left', padx=5)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset",
            command=self.reset_simulation,
            width=10
        )
        self.reset_btn.pack(side='left', padx=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = tk.Label(
            control_frame,
            textvariable=self.status_var,
            anchor='w',
            width=50
        )
        status_label.pack(side='left', padx=10)
    
    def add_initial_fire(self, count=5):
        """Add initial fire to the map."""
        empty_cells = list(zip(*np.where(self.grid_data == 0)))
        for _ in range(min(count, len(empty_cells))):
            i, j = empty_cells[np.random.randint(len(empty_cells))]
            if (i, j) != tuple(self.robot_pos) and (i, j) not in self.exits:
                self.grid_data[i, j] = 2
    
    def toggle_simulation(self):
        """Toggle simulation running state."""
        self.running = not self.running
        self.start_btn.config(text="⏸ Pause" if self.running else "▶ Start")
        if self.running:
            self.run_simulation()
    
    def run_simulation(self):
        """Run simulation steps."""
        if not self.running:
            return
            
        self.step_simulation()
        self.root.after(500, self.run_simulation)
    
    def step_simulation(self):
        """Perform one simulation step."""
        self.step_count += 1
        
        # Spread fire
        self.fire.step(self)
        
        # Simple random movement for demo
        # In a real implementation, this would use pathfinding
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        np.random.shuffle(directions)
        
        for di, dj in directions:
            ni, nj = self.robot_pos[0] + di, self.robot_pos[1] + dj
            if (0 <= ni < self.grid_data.shape[0] and 
                0 <= nj < self.grid_data.shape[1] and 
                self.grid_data[ni, nj] != 1):  # Not a wall
                self.robot_pos = [ni, nj]
                break
        
        # Check win/lose conditions
        if tuple(self.robot_pos) in self.exits:
            self.status_var.set(f"Escaped in {self.step_count} steps!")
            self.running = False
            self.start_btn.config(state='disabled')
        elif self.grid_data[tuple(self.robot_pos)] == 2:
            self.status_var.set("Robot caught in fire!")
            self.running = False
            self.start_btn.config(state='disabled')
        else:
            self.status_var.set(f"Step: {self.step_count}")
        
        self.update_display()
    
    def reset_simulation(self):
        """Reset the simulation."""
        self.running = False
        self.step_count = 0
        self.robot_pos = list(self.start_pos)
        
        # Reset grid (keep walls, clear fire)
        self.grid_data[self.grid_data == 2] = 0
        
        # Add initial fire
        self.add_initial_fire(5)
        
        # Update UI
        self.start_btn.config(text="▶ Start", state='normal')
        self.status_var.set("Ready")
        self.update_display()
    
    def update_display(self):
        """Update the canvas display."""
        self.canvas.delete('all')
        
        cell_size = 20
        
        # Draw grid
        for i in range(self.grid_data.shape[0]):
            for j in range(self.grid_data.shape[1]):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                if self.grid_data[i, j] == 1:  # Wall
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black')
                elif self.grid_data[i, j] == 2:  # Fire
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='red')
                elif (i, j) in self.exits:  # Exit
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='green')
                
                # Draw grid lines
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='gray')
        
        # Draw robot
        x1 = self.robot_pos[1] * cell_size + 2
        y1 = self.robot_pos[0] * cell_size + 2
        x2 = x1 + cell_size - 4
        y2 = y1 + cell_size - 4
        self.canvas.create_oval(x1, y1, x2, y2, fill='blue')
        
        self.canvas.update()

class FireSpread:
    def __init__(self, prob=0.1):
        self.prob = prob
    
    def step(self, sim):
        """Spread fire to adjacent cells."""
        new_fires = []
        
        for i in range(sim.grid_data.shape[0]):
            for j in range(sim.grid_data.shape[1]):
                if sim.grid_data[i, j] == 2:  # Fire
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4 directions
                        ni, nj = i + di, j + dj
                        if (0 <= ni < sim.grid_data.shape[0] and 
                            0 <= nj < sim.grid_data.shape[1] and 
                            sim.grid_data[ni, nj] == 0 and  # Empty
                            (ni, nj) != tuple(sim.robot_pos) and  # Not robot
                            (ni, nj) not in sim.exits and  # Not exit
                            np.random.random() < self.prob):
                            new_fires.append((ni, nj))
        
        # Add new fires
        for i, j in new_fires:
            sim.grid_data[i, j] = 2

def main():
    # Create the main window
    root = tk.Tk()
    
    # Path to your map file
    map_path = os.path.join("data", "building_map.txt")
    
    try:
        # Create and run the simulation
        sim = SimpleSimulation(root, map_path)
        root.mainloop()
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        error_window = tk.Toplevel(root)
        error_window.title("Error")
        tk.Label(error_window, text=error_msg, justify='left').pack(padx=20, pady=20)
        tk.Button(error_window, text="OK", command=root.quit).pack(pady=10)
        root.mainloop()

if __name__ == "__main__":
    main()
