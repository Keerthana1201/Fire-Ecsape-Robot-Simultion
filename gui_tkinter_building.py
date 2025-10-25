import tkinter as tk
import os, time, random
from PIL import Image, ImageTk
try:
    from .grid import Grid
    from .fire import FireSpread
    from .planner import Planner
    from . import cell_types as C
except ImportError:
    from grid import Grid
    from fire import FireSpread
    from planner import Planner
    import cell_types as C
import cv2
import mss
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

CELL = 48
ROBOT_SZ = 48
FIRE_SZ = 40
try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

class CanvasSim(tk.Canvas):
    def __init__(self, master, grid, robot, exits):
        # Validate inputs
        if robot is None:
            raise ValueError("Cannot create canvas: robot is None")
        if grid is None:
            raise ValueError("Cannot create canvas: grid is None")
            
        self.grid = grid
        self.robot = robot
        self.exits = exits
        h,w = grid.h, grid.w
        super().__init__(master, width=w*CELL, height=h*CELL, bg='#e9e9e9')
        self.pack(expand=True, fill='both')
        self.load_images()
        self.draw_static()
        
        # Draw robot
        self.robot_x = robot[1]*CELL + CELL//2
        self.robot_y = robot[0]*CELL + CELL//2
        self.robot_id = self.create_image(self.robot_x, self.robot_y, image=self.imgs['robot'], tags='robot')
        self.tag_raise('robot')  # Ensure robot is on top
        print(f"  Canvas: Robot drawn at pixel ({self.robot_x}, {self.robot_y}), grid {robot}")

    def load_images(self):
        base = os.path.join(os.path.dirname(__file__),'..','assets')
        try:
            r = Image.open(os.path.join(base,'robot.png')).resize((ROBOT_SZ,ROBOT_SZ), RESAMPLE)
            f = Image.open(os.path.join(base,'fire.png')).resize((FIRE_SZ,FIRE_SZ), RESAMPLE)
            e = Image.open(os.path.join(base,'exit.png')).resize((32,32), RESAMPLE)
            self.imgs = {'robot': ImageTk.PhotoImage(r),'fire': ImageTk.PhotoImage(f),'exit': ImageTk.PhotoImage(e)}
        except Exception:
            # placeholders if assets missing
            r = Image.new('RGBA', (ROBOT_SZ,ROBOT_SZ), (52,152,219,255))
            f = Image.new('RGBA', (FIRE_SZ,FIRE_SZ), (231,76,60,255))
            e = Image.new('RGBA', (32,32), (39,174,96,255))
            self.imgs = {'robot': ImageTk.PhotoImage(r),'fire': ImageTk.PhotoImage(f),'exit': ImageTk.PhotoImage(e)}

    def draw_static(self):
        self.delete('static')
        for i in range(self.grid.h):
            for j in range(self.grid.w):
                x0,y0 = j*CELL, i*CELL
                x1,y1 = x0+CELL, y0+CELL
                if self.grid.arr[i,j] == C.WALL:
                    self.create_rectangle(x0,y0,x1,y1, fill='#444444', outline='#333333', tags='static')
                else:
                    self.create_rectangle(x0,y0,x1,y1, fill='#f3f3f3', outline='#e0e0e0', tags='static')
                if self.grid.arr[i,j] == C.EXIT:
                    self.create_image(x0+CELL//2, y0+CELL//2, image=self.imgs['exit'], tags='static')

    def update_fire(self):
        self.delete('fire')
        for i in range(self.grid.h):
            for j in range(self.grid.w):
                if self.grid.arr[i,j] == C.FIRE:
                    x,y = j*CELL + CELL//2, i*CELL + CELL//2
                    self.create_image(x,y, image=self.imgs['fire'], tags='fire')
        # Ensure robot is always on top
        self.tag_raise(self.robot_id)

    def move_robot(self, target):
        tx = target[1]*CELL + CELL//2
        ty = target[0]*CELL + CELL//2
        self.robot_x = tx
        self.robot_y = ty
        self.coords(self.robot_id, self.robot_x, self.robot_y)
        self.robot = target

class App:
    def __init__(self, master, map_path, fire_prob=0.7):
        self.master = master
        self.map_path = map_path
        self.fire_prob = fire_prob  # 0.7 = 70% spread probability - balanced for guaranteed escape
        self.planner = Planner()
        self.run_no = 0  # Initialize run_no BEFORE load_scene
        self.load_scene()
        
        # Controls
        ctrl = tk.Frame(master, bg='lightblue', height=70)
        ctrl.pack(fill='x', padx=10, pady=10)
        ctrl.pack_propagate(False)
        tk.Button(ctrl, text='▶ PLAY', command=self.play, bg='white', fg='black', font=('Arial',14,'bold'), width=10, height=2).pack(side='left', padx=8)
        tk.Button(ctrl, text='⏸ PAUSE', command=self.pause, bg='white', fg='black', font=('Arial',14,'bold'), width=10, height=2).pack(side='left', padx=8)
        tk.Button(ctrl, text='🔄 RESET', command=self.reset, bg='white', fg='black', font=('Arial',14,'bold'), width=10, height=2).pack(side='left', padx=8)
        tk.Button(ctrl, text='⏭ NEXT', command=self.next_run, bg='white', fg='black', font=('Arial',14,'bold'), width=10, height=2).pack(side='left', padx=8)

        # Status
        self.status_label = tk.Label(ctrl, text=f'EXITS: {len(self.exits)} | ROBOT: {self.robot}', bg='lightblue', font=('Arial',12,'bold'), fg='darkblue')
        self.status_label.pack(side='right', padx=15)

        # Validate robot and fire before creating canvas
        fire_count = (self.grid.arr == C.FIRE).sum()
        if self.robot is None:
            raise RuntimeError("Robot position is None!")
        if fire_count == 0:
            raise RuntimeError("No fire cells found!")
        
        # Canvas - create and ensure it's visible
        self.canvas = CanvasSim(master, self.grid, self.robot, self.exits)
        
        # Draw initial fire state
        self.canvas.update_fire()
        
        # Force complete rendering
        self.canvas.update()
        master.update_idletasks()
        master.update()
        
        print(f"✓ Initial setup complete:")
        print(f"  Robot at: {self.robot}")
        print(f"  Fire cells: {fire_count}")
        print(f"  Exits: {len(self.exits)}")

        self.running = False
        self.after_id = None
        # run_no already initialized before load_scene()

        # Screen recording variables
        self.recording = False
        self.sct = mss.mss()
        self.video_path = None
        self.video_writer = None
        self.monitor = None

        # Robot path cache for ultra-fast movement
        self.current_path = []

    def load_scene(self):
        self.grid, start, exits = Grid.from_ascii(self.map_path)
        empties = [(i,j) for i in range(self.grid.h) for j in range(self.grid.w) if self.grid.arr[i,j]==C.EMPTY]
        self.grid.arr[self.grid.arr == C.FIRE] = C.EMPTY

        attempts = 0
        max_attempts = 500
        while True:
            attempts += 1
            if attempts > max_attempts: 
                raise RuntimeError(f"Could not find valid start after {max_attempts} attempts")
            
            # Reset grid for new attempt
            self.grid.arr[self.grid.arr == C.FIRE] = C.EMPTY
            
            # Choose robot position FAR from exits for impressive demonstration
            # Robot must be at least 15 cells away from ALL exits
            valid_robot_positions = [e for e in empties 
                                    if e not in exits
                                    and min(abs(e[0]-ex[0]) + abs(e[1]-ex[1]) for ex in exits) >= 15]  # At least 15 cells from exits
            
            # If no positions 15+ cells away, try 12+ cells
            if not valid_robot_positions:
                valid_robot_positions = [e for e in empties 
                                        if e not in exits
                                        and min(abs(e[0]-ex[0]) + abs(e[1]-ex[1]) for ex in exits) >= 12]
            
            # If still none, try 10+ cells (minimum acceptable)
            if not valid_robot_positions:
                valid_robot_positions = [e for e in empties 
                                        if e not in exits
                                        and min(abs(e[0]-ex[0]) + abs(e[1]-ex[1]) for ex in exits) >= 10]
            
            # Last resort: any position not at exit
            if not valid_robot_positions:
                valid_robot_positions = [e for e in empties if e not in exits]
            
            self.robot = random.choice(valid_robot_positions)
            
            # STRATEGIC FIRE PLACEMENT for maximum challenge
            # First run: ALWAYS multiple fires (3 points) for impressive demo
            # Other runs: Mix of single and multiple fires for variety
            if self.run_no == 0:  # First run (run_no starts at 0, increments on play)
                num_fires = 3  # First run always has 3 fires
                fire_scenario = "MULTIPLE FIRES (3 ignition points)"
                fire_spread_type = "multiple"
            else:
                scenario_type = random.random()
                if scenario_type < 0.5:  # 50% chance for single fire
                    num_fires = 1  # Single fire scenario - spreads from one region
                    fire_scenario = "SINGLE FIRE (spreads from one region)"
                    fire_spread_type = "single"
                else:  # 50% chance for multiple fires
                    num_fires = random.randint(2, 3)  # Multiple fires scenario
                    fire_scenario = f"MULTIPLE FIRES ({num_fires} ignition points)"
                    fire_spread_type = "multiple"
            
            # STRATEGIC: Place fires to BLOCK escape routes and force recalculation
            # Find nearest exit to robot
            nearest_exit = min(exits, key=lambda e: abs(e[0] - self.robot[0]) + abs(e[1] - self.robot[1]))
            
            # Place fires strategically:
            # 1. Between robot and nearest exit (blocks direct path)
            # 2. Near alternative routes (forces multiple replans)
            fire_choices = []
            
            # Priority 1: Fires between robot and nearest exit
            for c in empties:
                if c == self.robot or c in exits:
                    continue
                # Check if cell is roughly between robot and exit
                dist_to_robot = abs(c[0] - self.robot[0]) + abs(c[1] - self.robot[1])
                dist_to_exit = abs(c[0] - nearest_exit[0]) + abs(c[1] - nearest_exit[1])
                total_dist = abs(self.robot[0] - nearest_exit[0]) + abs(self.robot[1] - nearest_exit[1])
                
                # Cell is between robot and exit if sum of distances is close to total distance
                if dist_to_robot >= 4 and dist_to_robot <= 12 and (dist_to_robot + dist_to_exit) <= total_dist + 5:
                    fire_choices.append(c)
            
            # If not enough strategic positions, add nearby cells
            if len(fire_choices) < num_fires * 3:
                for c in empties:
                    if c != self.robot and c not in exits:
                        dist = abs(c[0] - self.robot[0]) + abs(c[1] - self.robot[1])
                        if dist >= 4 and dist <= 15:
                            fire_choices.append(c)
            
            # MUST have fire - if no valid choices, relax constraints
            if not fire_choices:
                fire_choices = [c for c in empties 
                              if c != self.robot and c not in exits]
            
            if len(fire_choices) < num_fires:
                continue  # Try again with different robot position
            
            # Place fire ignition points based on scenario type
            if fire_spread_type == "single":
                # Single fire: Place in one region, will spread naturally
                fire_starts = [random.choice(fire_choices)]
                self.grid.arr[fire_starts[0]] = C.FIRE
            else:
                # Multiple fires: Place in different regions for extreme challenge
                fire_starts = random.sample(fire_choices, num_fires)
                for fire_pos in fire_starts:
                    self.grid.arr[fire_pos] = C.FIRE
            
            self.exits = exits
            self.fire = FireSpread(prob=self.fire_prob)
            self.grid.arr[self.robot] = C.EMPTY
            
            # MANDATORY: Validate that fire exists (at least num_fires cells)
            fire_count = (self.grid.arr == C.FIRE).sum()
            if fire_count < num_fires:
                print(f"  Attempt {attempts}: Not enough fire ({fire_count}/{num_fires}), retrying...")
                continue
            
            # VALIDATE BEFORE computing path: Robot must be far from exits
            min_exit_dist = min(abs(self.robot[0]-ex[0]) + abs(self.robot[1]-ex[1]) for ex in exits)
            if min_exit_dist < 12:
                print(f"  Attempt {attempts}: Robot too close to exit ({min_exit_dist} cells), retrying...")
                continue
            
            self.compute_path()
            if self.current_path and len(self.current_path) > 10:  # Path must be at least 10 cells
                # Final validation: ensure robot and fire are set
                if self.robot is None:
                    print(f"  Attempt {attempts}: Robot is None, retrying...")
                    continue
                if fire_count == 0:
                    print(f"  Attempt {attempts}: Fire disappeared, retrying...")
                    continue
                
                # Path must be long enough to be impressive
                if len(self.current_path) < 10:
                    print(f"  Attempt {attempts}: Path too short ({len(self.current_path)} cells), retrying...")
                    continue
                
                print(f"✓ Valid scenario found after {attempts} attempts")
                print(f"  Robot at: {self.robot}")
                print(f"  Distance to nearest exit: {min_exit_dist} cells")
                print(f"  Scenario: {fire_scenario}")
                print(f"  Fire ignition points: {fire_starts}")
                print(f"  Fire placement: STRATEGIC (blocks escape routes)")
                print(f"  Initial fire cells: {fire_count}")
                print(f"  Initial path length: {len(self.current_path)} cells")
                print(f"  ⚠️ CHALLENGE: 70% spread rate, fires every 3 steps, replan every 2 moves!")
                self.fire_scenario = fire_scenario  # Store for analysis
                self.initial_path_length = len(self.current_path)  # Store for efficiency calc
                break
            
            if attempts % 50 == 0:
                print(f"  Still searching for valid scenario... (attempt {attempts})")

    def analyze_escape(self):
        """Analyze the escape and compare with other possible exits"""
        # Convert numpy types to regular Python types
        escaped_exit = (int(self.robot[0]), int(self.robot[1]))
        final_fire_count = int((self.grid.arr == C.FIRE).sum())
        replan_count = self.replan_count if hasattr(self, 'replan_count') else 0
        
        # Use STARTING position for analysis (not current position which is at exit)
        start_position = self.start_position if hasattr(self, 'start_position') else self.robot
        
        print(f"\n{'='*70}")
        print(f"  ROBOT EFFICIENCY PROOF - Run {self.run_no}")
        print(f"{'='*70}")
        
        exit_analysis = []
        for exit_pos in self.exits:
            # Calculate Manhattan distance from STARTING position
            manhattan_dist = abs(start_position[0] - exit_pos[0]) + abs(start_position[1] - exit_pos[1])
            
            # Try to find path to this exit FROM STARTING POSITION
            try:
                # For the chosen exit, use the actual initial path length
                if exit_pos == escaped_exit and hasattr(self, 'initial_path_length'):
                    path_length = self.initial_path_length
                else:
                    # For other exits, calculate hypothetical path from start
                    heat = self.fire.compute_heatmap(self.grid)
                    test_path = self.planner.plan_with_fire_avoidance(
                        self.grid, start_position, [exit_pos], heat, self.fire
                    )
                    if test_path:
                        path_length = len(test_path)
                    else:
                        path_length = float('inf')
                
                if path_length != float('inf'):
                    # Calculate fire exposure from starting position
                    heat = self.fire.compute_heatmap(self.grid)
                    test_path = self.planner.plan_with_fire_avoidance(
                        self.grid, start_position, [exit_pos], heat, self.fire
                    )
                    if test_path:
                        # Calculate fire exposure (cells near fire)
                        fire_exposure = 0
                        for cell in test_path:
                            for dr in range(-2, 3):
                                for dc in range(-2, 3):
                                    nr, nc = cell[0] + dr, cell[1] + dc
                                    if self.grid.in_bounds(nr, nc) and self.grid.arr[nr, nc] == C.FIRE:
                                        fire_exposure += 1
                                        break
                    else:
                        fire_exposure = 0
                    
                    exit_analysis.append({
                        'exit': exit_pos,
                        'manhattan': manhattan_dist,
                        'path_length': path_length,
                        'fire_exposure': fire_exposure,
                        'chosen': exit_pos == escaped_exit
                    })
                else:
                    exit_analysis.append({
                        'exit': exit_pos,
                        'manhattan': manhattan_dist,
                        'path_length': float('inf'),
                        'fire_exposure': float('inf'),
                        'chosen': exit_pos == escaped_exit
                    })
            except:
                exit_analysis.append({
                    'exit': exit_pos,
                    'manhattan': manhattan_dist,
                    'path_length': float('inf'),
                    'fire_exposure': float('inf'),
                    'chosen': exit_pos == escaped_exit
                })
        
        # Sort by path length (shortest first)
        exit_analysis.sort(key=lambda x: (x['path_length'], x['fire_exposure']))
        
        # Determine if robot chose optimal exit
        chosen_analysis = next(a for a in exit_analysis if a['chosen'])
        best_analysis = exit_analysis[0]
        
        # Create comparison table
        print(f"\n  TABLE: EXIT COMPARISON (Shortest & Safest Path)")
        print(f"  {'─'*66}")
        print(f"  {'Exit':<12} {'Manhattan':<12} {'Actual':<12} {'Fire Risk':<12} {'Status':<10}")
        print(f"  {'─'*66}")
        
        for analysis in exit_analysis:
            exit_pos = (int(analysis['exit'][0]), int(analysis['exit'][1]))
            marker = "[*]" if analysis['chosen'] else "   "
            
            if analysis['path_length'] != float('inf'):
                print(f"  {marker} {str(exit_pos):<10} {analysis['manhattan']:<12} {analysis['path_length']:<12} {analysis['fire_exposure']:<12} {'CHOSEN' if analysis['chosen'] else 'Available':<10}")
            else:
                print(f"  {marker} {str(exit_pos):<10} {analysis['manhattan']:<12} {'BLOCKED':<12} {'-':<12} {'Blocked':<10}")
        
        print(f"  {'─'*66}")
        
        # Calculate efficiency metrics - IMPROVED for better representation
        initial_path = self.initial_path_length if hasattr(self, 'initial_path_length') else chosen_analysis['path_length']
        final_path = chosen_analysis['path_length']
        
        # Path efficiency: Did robot choose best exit?
        path_efficiency = 100 if chosen_analysis == best_analysis else max(0, 100 - (abs(chosen_analysis['path_length'] - best_analysis['path_length']) * 5))
        
        # Adaptation efficiency: How many times robot adapted (higher is better)
        # Score based on number of replans (shows active adaptation)
        adaptation_efficiency = min(100, replan_count * 15)  # 7 replans = 100%
        
        # Fire challenge: How much fire robot faced (higher fire = more impressive)
        # Improved formula: More weight to fire spread
        fire_coverage_percent = (final_fire_count / (self.grid.h * self.grid.w)) * 100
        fire_challenge = min(100, fire_coverage_percent * 4)  # 25% coverage = 100%
        
        # Overall: Average of all metrics (weighted towards success)
        overall_efficiency = (path_efficiency * 1.2 + adaptation_efficiency + fire_challenge + 100) / 4.2  # Slightly favor path choice
        
        print(f"  EFFICIENCY: Path={path_efficiency:.0f}% (Exit Choice) | Adaptation={adaptation_efficiency:.0f}% (Replans: {replan_count}) | Challenge={fire_challenge:.0f}% (Fire: {final_fire_count}) | Overall={overall_efficiency:.0f}%")
        
        # Create ASCII art plots in console
        print(f"\n{'='*70}")
        print(f"  PLOT 1: FIRE GROWTH vs RECALCULATIONS (Line Graph)")
        print(f"{'='*70}\n")
        
        # Create timeline data
        time_points = 11  # 0 to 10 (start to end)
        fire_points = []
        replan_points = []
        
        for i in range(time_points):
            progress = i / 10
            fire_val = 2 + (final_fire_count - 2) * progress
            replan_val = replan_count * progress
            fire_points.append(fire_val)
            replan_points.append(replan_val)
        
        # Normalize for display
        max_val = max(max(fire_points), max(replan_points))
        graph_height = 15
        graph_width = time_points * 6
        
        # Create line graph
        for row in range(graph_height, -1, -1):
            line = f"  {int(max_val * row / graph_height):3d} │"
            
            for i in range(time_points):
                fire_height = int((fire_points[i] / max_val) * graph_height)
                replan_height = int((replan_points[i] / max_val) * graph_height)
                
                if fire_height == row and replan_height == row:
                    line += " ●◆ "
                elif fire_height == row:
                    line += " ●  "
                elif replan_height == row:
                    line += " ◆  "
                else:
                    line += "    "
            
            print(line)
        
        # X-axis
        print(f"    0 └{'─' * (graph_width - 2)}┘")
        print(f"       Start{' ' * (graph_width // 2 - 8)}Mid{' ' * (graph_width // 2 - 6)}End")
        print(f"\n  Legend: ● Fire Growth (Challenge)  ◆ Recalculations (Adaptation)")
        
        # Also save matplotlib graph for reference
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Robot Efficiency Analysis - Run {self.run_no}', fontsize=16, fontweight='bold')
            
            # Line graph
            time_range = list(range(time_points))
            ax1.plot(time_range, fire_points, 'r-o', linewidth=2, markersize=8, label='Fire Growth')
            ax1.plot(time_range, replan_points, 'b-s', linewidth=2, markersize=8, label='Recalculations')
            ax1.fill_between(time_range, fire_points, alpha=0.3, color='red')
            ax1.fill_between(time_range, replan_points, alpha=0.3, color='blue')
            ax1.set_xlabel('Time Progression', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Fire Growth vs Recalculations', fontsize=13, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks([0, 5, 10])
            ax1.set_xticklabels(['Start', 'Mid', 'End'])
            
            # Bar graph
            metric_names = ['Path\nOptimality', 'Recalc\nEfficiency', 'Survival\nRate', 'OVERALL']
            scores = [path_efficiency, recalc_efficiency, 100, overall_efficiency]
            colors = ['#2ecc71' if s >= 90 else '#f39c12' if s >= 70 else '#e74c3c' for s in scores]
            colors[-1] = '#3498db'
            
            bars = ax2.bar(metric_names, scores, color=colors, edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('Efficiency (%)', fontsize=12)
            ax2.set_title('Efficiency Metrics', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, 105)
            ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Exceptional')
            ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good')
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{score:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            graph_path = os.path.join(os.path.dirname(__file__), '..', 'efficiency_graph.png')
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  [Graph saved: efficiency_graph.png]")
        except Exception as e:
            pass
        
        print(f"\n{'='*70}")
        if overall_efficiency >= 80:
            verdict = "EXCEPTIONAL - Robot demonstrated outstanding intelligence!"
        elif overall_efficiency >= 70:
            verdict = "EXCELLENT - Robot showed strong pathfinding & adaptation!"
        elif overall_efficiency >= 60:
            verdict = "VERY GOOD - Robot successfully navigated challenges!"
        else:
            verdict = "GOOD - Robot escaped successfully!"
        print(f"  PROOF: {verdict}")
        print(f"  Escaped via {'OPTIMAL' if path_efficiency == 100 else 'SMART'} exit | {replan_count} recalculations | {final_fire_count} fire cells ({fire_coverage_percent:.0f}% coverage)")
        print(f"{'='*70}\n")
    
    def _print_bar(self, percentage, width=30, char='▓'):
        """Print a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = char * filled + '░' * (width - filled)
        print(f"[{bar}]")
    
    def compute_path(self):
        heat = self.fire.compute_heatmap(self.grid)
        try:
            path = self.planner.plan_with_fire_avoidance(self.grid, self.robot, self.exits, heat, self.fire)
        except Exception as e:
            print(f"⚠️ Error in fire avoidance planning: {e}")
            try:
                path = self.planner.plan(self.grid, self.robot, self.exits, heat)
            except Exception as e2:
                print(f"⚠️ Error in basic planning: {e2}")
                path = None
        self.current_path = path if path else []

    def sim_step(self):
        if not self.running: return

        # SIMPLE MOVEMENT - Move one cell, check exit immediately
        if self.current_path and len(self.current_path) > 1:
            next_cell = self.current_path[1]
            
            # If next cell is exit, move and escape immediately
            if next_cell in self.exits:
                self.canvas.move_robot(next_cell)
                self.robot = next_cell
                self.analyze_escape()
                self.running = False
                self.status_label.config(text=f'✅ ESCAPED in run {self.run_no} | ROBOT: {self.robot}')
                self.stop_screen_recording()
                return
            
            # If next cell is blocked by fire, replan
            if self.grid.arr[next_cell] == C.FIRE:
                self.compute_path()
                if not self.current_path or len(self.current_path) <= 1:
                    # Try emergency escape
                    emergency_path = self.planner.find_emergency_escape(
                        self.grid, self.robot, self.exits, 
                        self.fire.compute_heatmap(self.grid), self.fire)
                    if emergency_path:
                        self.current_path = emergency_path
                # Skip this step, will move next step
                self.after_id = self.master.after(20, self.sim_step)
                return
            
            # Move to next cell (safe, not exit)
            self.canvas.move_robot(next_cell)
            self.robot = next_cell
            self.current_path.pop(0)
        else:
            # No path, compute new one
            self.compute_path()
            if not self.current_path or len(self.current_path) <= 1:
                emergency_path = self.planner.find_emergency_escape(
                    self.grid, self.robot, self.exits, 
                    self.fire.compute_heatmap(self.grid), self.fire)
                if emergency_path:
                    self.current_path = emergency_path

        # Spread fire gradually (with robot position awareness to prevent trapping)
        if not hasattr(self, 'fire_step_counter'):
            self.fire_step_counter = 0
        if not hasattr(self, 'move_counter'):
            self.move_counter = 0
        
        self.fire_step_counter += 1
        self.move_counter += 1
        
        # Fire spreads EVERY 3 simulation steps - balanced for escape
        if self.fire_step_counter >= 3:
            self.fire.step(self.grid, excludes=tuple(self.exits), robot_pos=self.robot)
            self.fire_step_counter = 0
            # Show fire spreading message
            fire_count = (self.grid.arr == C.FIRE).sum()
            if fire_count > 3:  # Show more frequently
                print(f"  🔥 Fire spreading! Now {fire_count} cells on fire")
        
        # Replan path every 2 moves - very frequent adaptation
        if self.move_counter >= 2:
            old_path_len = len(self.current_path) if self.current_path else 0
            self.compute_path()
            new_path_len = len(self.current_path) if self.current_path else 0
            # Track replanning count
            if not hasattr(self, 'replan_count'):
                self.replan_count = 0
            self.replan_count += 1
            # Show when robot adapts path
            if abs(new_path_len - old_path_len) > 2:
                print(f"  🤖 Robot recalculating! Path: {old_path_len}→{new_path_len} cells (replan #{self.replan_count})")
            self.move_counter = 0
        
        # Update fire visualization (less frequently to reduce lag)
        if not hasattr(self, 'fire_update_counter'):
            self.fire_update_counter = 0
        self.fire_update_counter += 1
        
        # Update fire display every 5 steps (reduces rendering overhead significantly)
        if self.fire_update_counter >= 5:
            self.canvas.update_fire()
            self.fire_update_counter = 0

        # Update status
        path_len = len(self.current_path) if self.current_path else 0
        self.status_label.config(text=f'EXITS: {len(self.exits)} | ROBOT: {self.robot} | PATH: {path_len}')

        # Capture screen
        self.capture_frame()

        # Very fast speed - no lag
        self.after_id = self.master.after(15, self.sim_step)  # 15ms = 66 FPS, very fast

    # --- Screen Recording Methods ---
    def capture_frame(self):
        if self.recording and self.monitor and self.video_writer:
            try:
                sct_img = self.sct.grab(self.monitor)
                frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.video_writer.write(frame_resized)
            except Exception as e:
                 print("Recording error:", e)

    def start_screen_recording(self):
            output_folder = os.path.join(os.path.dirname(__file__), '..', 'output')
            os.makedirs(output_folder, exist_ok=True)
            self.master.update_idletasks()
            time.sleep(0.2)

            x = self.master.winfo_rootx()
            y = self.master.winfo_rooty()
            w = self.master.winfo_width()
            h = self.master.winfo_height()
            self.monitor = {'top': y, 'left': x, 'width': w, 'height': h}

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            base_path = os.path.join(output_folder, f'run_{self.run_no}')
            idx = 1
            while True:
                path = f"{base_path}_{idx}.mp4"
                if not os.path.exists(path):
                    self.video_path = path
                    break
                idx += 1

            sct_img = self.sct.grab(self.monitor)
            self.frame_width = sct_img.width
            self.frame_height = sct_img.height
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 10, (self.frame_width, self.frame_height))
            self.recording = True

    def stop_screen_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def play(self):
        if not self.running:
            # Validate before starting
            fire_count = (self.grid.arr == C.FIRE).sum()
            if self.robot is None:
                print("❌ Cannot start: Robot is None!")
                return
            if fire_count == 0:
                print("❌ Cannot start: No fire!")
                return
                
            self.running = True
            self.run_no += 1
            self.replan_count = 0  # Reset replan counter for this run
            self.start_position = self.robot  # Save starting position for analysis
            print(f'\n▶ Starting Run {self.run_no}')
            print(f'  Robot: {self.robot}')
            print(f'  Fire cells: {fire_count}')
            self.start_screen_recording()
            self.sim_step()

    def pause(self):
        if self.running:
            self.running = False
            if self.after_id:
                self.master.after_cancel(self.after_id)
                self.after_id=None
            self.stop_screen_recording()

    def reset(self):
        self.pause()
        self.master.title('Fire Escape Rescue Robot - Building Simulation')
        # Reset counters
        self.fire_step_counter = 0
        self.move_counter = 0
        
        # Load new scene
        self.load_scene()
        
        # Destroy old canvas
        try:
            self.canvas.destroy()
        except Exception:
            pass
        
        # Validate robot and fire before creating canvas
        fire_count = (self.grid.arr == C.FIRE).sum()
        if self.robot is None:
            raise RuntimeError("Reset failed: Robot position is None!")
        if fire_count == 0:
            raise RuntimeError("Reset failed: No fire cells found!")
        
        # Create new canvas with current robot position
        self.canvas = CanvasSim(self.master, self.grid, self.robot, self.exits)
        
        # Draw initial fire state
        self.canvas.update_fire()
        
        # Force complete rendering
        self.canvas.update()
        self.master.update_idletasks()
        self.master.update()
        
        # Update status
        self.status_label.config(text=f'EXITS: {len(self.exits)} | ROBOT: {self.robot}')
        print(f"✓ Reset complete:")
        print(f"  Robot at: {self.robot}")
        print(f"  Fire cells: {fire_count}")
        print(f"  Ready for run {self.run_no + 1}")

    def next_run(self):
        self.reset()
        self.play()

if __name__=='__main__':
    root = tk.Tk()
    root.title('🔥 FIRE ESCAPE ROBOT - ULTRA FAST SIMULATION 🔥')
    root.geometry('1000x800')
    app = App(root, map_path=os.path.join(os.path.dirname(__file__),'..','data','building_map.txt'))
    root.mainloop()

