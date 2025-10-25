import tkinter as tk
import os, time, random
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# Import our new components
from risk_aware_planner import RiskAwarePlanner
from fire_risk import FireRiskAnalyzer
from risk_visualization import create_risk_overlay, draw_risk_legend

try:
    from .grid import Grid
    from .fire import FireSpread
    from . import cell_types as C
except ImportError:
    from grid import Grid
    from fire import FireSpread
    import cell_types as C

CELL = 48
ROBOT_SZ = 48
FIRE_SZ = 40

class RiskAwareCanvas(tk.Canvas):
    def __init__(self, master, grid, robot, exits):
        self.grid = grid
        self.robot = robot
        self.exits = exits
        h, w = grid.h, grid.w
        
        super().__init__(master, width=w*CELL, height=h*CELL, bg='#e9e9e9')
        self.pack(expand=True, fill='both')
        
        # Initialize risk analyzer
        self.risk_analyzer = FireRiskAnalyzer()
        self.risk_overlay = None
        self.risk_overlay_id = None
        self.show_risk = True  # Toggle for risk visualization
        
        self.load_images()
        self.draw_static()
        
        # Draw robot
        self.robot_x = robot[1]*CELL + CELL//2
        self.robot_y = robot[0]*CELL + CELL//2
        self.robot_id = self.create_image(self.robot_x, self.robot_y, 
                                        image=self.imgs['robot'], tags='robot')
        self.tag_raise('robot')  # Ensure robot is on top
        
        # Add risk legend
        self.legend_id = draw_risk_legend(self, 10, 10)
    
    def load_images(self):
        """Load and cache images."""
        self.imgs = {
            'wall': self.create_rectangle(0, 0, CELL, CELL, fill='#333333', outline=''),
            'fire': self.create_oval(CELL//2-FIRE_SZ//2, CELL//2-FIRE_SZ//2,
                                   CELL//2+FIRE_SZ//2, CELL//2+FIRE_SZ//2,
                                   fill='#ff3300', outline='#ff9900', width=2),
            'exit': self.create_rectangle(CELL//4, CELL//4, 3*CELL//4, 3*CELL//4,
                                        fill='#00cc00', outline='#006600', width=2),
            'robot': self.create_oval(CELL//2-ROBOT_SZ//2, CELL//2-ROBOT_SZ//2,
                                    CELL//2+ROBOT_SZ//2, CELL//2+ROBOT_SZ//2,
                                    fill='#3366ff', outline='#003399', width=2)
        }
        
        # Hide template images
        for img in self.imgs.values():
            self.itemconfig(img, state='hidden')
    
    def draw_static(self):
        """Draw static elements (walls, exits)."""
        self.delete('static')
        
        # Draw walls and exits
        for i in range(self.grid.h):
            for j in range(self.grid.w):
                x1, y1 = j * CELL, i * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                
                if self.grid.arr[i, j] == C.WALL:
                    self.create_rectangle(x1, y1, x2, y2, 
                                        fill='#333333', outline='', tags='wall')
                elif (i, j) in self.exits:
                    self.create_rectangle(x1 + CELL//4, y1 + CELL//4,
                                        x2 - CELL//4, y2 - CELL//4,
                                        fill='#00cc00', outline='#006600',
                                        width=2, tags='exit')
    
    def update_fire(self, fire_cells):
        """Update fire visualization."""
        self.delete('fire')
        
        for i, j in fire_cells:
            x = j * CELL + CELL // 2
            y = i * CELL + CELL // 2
            self.create_image(x, y, image=self.imgs['fire'], tags='fire')
        
        # Update risk visualization
        self.update_risk_overlay()
    
    def update_risk_overlay(self):
        """Update the risk visualization overlay."""
        if not self.show_risk:
            if self.risk_overlay_id is not None:
                self.delete(self.risk_overlay_id)
                self.risk_overlay_id = None
            return
        
        # Get current fire state and update risk map
        fire_map = (self.grid.arr == C.FIRE).astype(int)
        risk_map = self.risk_analyzer.update_risk(fire_map)
        
        # Create and update overlay
        self.risk_overlay = create_risk_overlay(risk_map)
        
        if self.risk_overlay:
            # Convert to PhotoImage
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(self.risk_overlay)
            
            # Update or create overlay
            if self.risk_overlay_id is not None:
                self.delete(self.risk_overlay_id)
            
            self.risk_overlay_id = self.create_image(0, 0, image=photo, anchor='nw', tags='risk')
            self.lower('risk')  # Move below robot and fire
            
            # Keep reference to prevent garbage collection
            self.risk_photo = photo
    
    def toggle_risk_visualization(self):
        """Toggle risk visualization on/off."""
        self.show_risk = not self.show_risk
        if self.show_risk:
            self.update_risk_overlay()
        elif self.risk_overlay_id is not None:
            self.delete(self.risk_overlay_id)
            self.risk_overlay_id = None

class RiskAwareApp:
    def __init__(self, master, map_path, fire_prob=0.7):
        self.master = master
        self.map_path = map_path
        self.fire_prob = fire_prob
        self.planner = RiskAwarePlanner()
        self.load_scene()
        
        # Setup UI
        self.setup_ui()
        
        # Start simulation
        self.running = False
        self.step_count = 0
        self.start_time = time.time()
        
        # Start with a short delay to ensure everything is initialized
        self.master.after(100, self.sim_step)
    
    def setup_ui(self):
        """Set up the user interface."""
        # Control panel
        ctrl = tk.Frame(self.master, bg='lightblue', height=70)
        ctrl.pack(fill='x', padx=10, pady=10)
        ctrl.pack_propagate(False)
        
        # Control buttons
        tk.Button(ctrl, text='▶ PLAY', command=self.play, 
                 bg='white', fg='black', font=('Arial', 12, 'bold'), 
                 width=10).pack(side='left', padx=5)
        
        tk.Button(ctrl, text='⏸ PAUSE', command=self.pause, 
                 bg='white', fg='black', font=('Arial', 12, 'bold'), 
                 width=10).pack(side='left', padx=5)
        
        tk.Button(ctrl, text='🔄 RESET', command=self.reset, 
                 bg='white', fg='black', font=('Arial', 12, 'bold'), 
                 width=10).pack(side='left', padx=5)
        
        # Toggle risk visualization
        self.risk_btn = tk.Button(
            ctrl, text='⚠️ HIDE RISK', command=self.toggle_risk_visualization,
            bg='#ffcc00', fg='black', font=('Arial', 12, 'bold'), width=12
        )
        self.risk_btn.pack(side='right', padx=5)
        
        # Status display
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        status = tk.Label(ctrl, textvariable=self.status_var, 
                         bg='lightblue', font=('Arial', 10, 'bold'))
        status.pack(side='right', padx=10)
        
        # Create canvas
        self.canvas = RiskAwareCanvas(self.master, self.grid, self.robot, self.exits)
    
    def load_scene(self):
        """Load the map and initialize simulation state."""
        # Load grid from file
        self.grid = Grid.from_file(self.map_path)
        
        # Find robot and exits
        self.robot = None
        self.exits = []
        
        for i in range(self.grid.h):
            for j in range(self.grid.w):
                if self.grid.arr[i, j] == C.START:
                    self.robot = (i, j)
                    self.grid.arr[i, j] = C.EMPTY  # Clear start position
                elif self.grid.arr[i, j] == C.EXIT:
                    self.exits.append((i, j))
                    self.grid.arr[i, j] = C.EMPTY  # Clear exit positions
        
        if not self.robot:
            self.robot = (self.grid.h // 2, self.grid.w // 2)
        
        if not self.exits:
            # Add default exits at the corners
            self.exits = [
                (0, 0),
                (0, self.grid.w - 1),
                (self.grid.h - 1, 0),
                (self.grid.h - 1, self.grid.w - 1)
            ]
        
        # Initialize fire spread
        self.fire = FireSpread(prob=self.fire_prob)
        
        # Add initial fire
        for _ in range(min(3, self.grid.h * self.grid.w // 20)):
            while True:
                i = random.randint(0, self.grid.h - 1)
                j = random.randint(0, self.grid.w - 1)
                if (self.grid.arr[i, j] == C.EMPTY and 
                    (i, j) != self.robot and 
                    (i, j) not in self.exits):
                    self.grid.arr[i, j] = C.FIRE
                    break
        
        # Initialize path
        self.path = []
        self.path_index = 0
    
    def sim_step(self):
        """Perform one simulation step."""
        if not self.running:
            self.master.after(100, self.sim_step)
            return
        
        self.step_count += 1
        
        # Update fire spread
        self.fire.step(self.grid, excludes=[self.robot] + self.exits, 
                      robot_pos=self.robot)
        
        # Update fire visualization
        fire_cells = list(zip(*np.where(self.grid.arr == C.FIRE)))
        self.canvas.update_fire(fire_cells)
        
        # Check if robot is on fire
        if self.grid.arr[self.robot] == C.FIRE:
            self.status_var.set("Robot caught in fire!")
            self.running = False
            return
        
        # Check if robot reached an exit
        if self.robot in self.exits:
            self.status_var.set("Robot escaped successfully!")
            self.running = False
            return
        
        # Plan path if needed
        if not self.path or self.path_index >= len(self.path):
            self.plan_path()
        
        # Move robot along path
        if self.path and self.path_index < len(self.path):
            next_pos = self.path[self.path_index]
            
            # Check if next position is safe
            if (0 <= next_pos[0] < self.grid.h and 
                0 <= next_pos[1] < self.grid.w and 
                self.grid.arr[next_pos] != C.WALL):
                
                # Move robot
                self.robot = next_pos
                self.path_index += 1
                
                # Update robot position on canvas
                self.canvas.robot_x = self.robot[1] * CELL + CELL // 2
                self.canvas.robot_y = self.robot[0] * CELL + CELL // 2
                self.canvas.coords(self.canvas.robot_id, 
                                 self.canvas.robot_x, self.canvas.robot_y)
        
        # Update status
        self.status_var.set(f"Step: {self.step_count} | "
                          f"Position: {self.robot} | "
                          f"Risk: {self.planner.get_risk_category(self.robot).upper()}")
        
        # Continue simulation
        self.master.after(100, self.sim_step)
    
    def plan_path(self):
        """Plan a new path to the nearest exit."""
        # Get heatmap from fire simulation
        heatmap = self.fire.compute_heatmap(self.grid)
        
        # Find best path using risk-aware planner
        self.path = self.planner.plan(self.grid, self.robot, self.exits, heatmap)
        self.path_index = 0
    
    def play(self):
        """Start or resume simulation."""
        self.running = True
        if self.step_count == 0:
            self.sim_step()
    
    def pause(self):
        """Pause simulation."""
        self.running = False
    
    def reset(self):
        """Reset simulation to initial state."""
        self.running = False
        self.step_count = 0
        self.start_time = time.time()
        self.load_scene()
        
        # Recreate canvas
        self.canvas.destroy()
        self.canvas = RiskAwareCanvas(self.master, self.grid, self.robot, self.exits)
        self.canvas.pack(expand=True, fill='both')
    
    def toggle_risk_visualization(self):
        """Toggle risk visualization on/off."""
        self.canvas.toggle_risk_visualization()
        if self.canvas.show_risk:
            self.risk_btn.config(text="⚠️ HIDE RISK")
        else:
            self.risk_btn.config(text="👁️ SHOW RISK")
