from heat_aware_astar_improved import astar_heat
import numpy as np

def choose_best_exit(robot, exits, grid, fire, heatmap, fire_map):
    """Choose the best exit considering path safety and fire proximity."""
    best_exit = None
    best_score = float('inf')
    best_path = None
    
    for exit_pos in exits:
        # Calculate path with fire awareness
        path = astar_heat(
            grid, 
            robot, 
            exit_pos, 
            heatmap=heatmap, 
            allow_fire=False,  # First try without allowing fire
            fire_map=fire_map
        )
        
        if path:
            # Calculate safety score (lower is better)
            safety_score = 0
            fire_cells_in_path = 0
            
            for i, (r, c) in enumerate(path):
                # Check if this cell is on fire
                if grid.arr[r, c] == 2:  # Fire cell
                    safety_score += 1000  # Very high penalty
                    fire_cells_in_path += 1
                
                # Check surrounding fire density
                fire_count = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < grid.h and 0 <= nc < grid.w and 
                            grid.arr[nr, nc] == 2):
                            # Closer fire is more dangerous
                            fire_count += 1 / (1 + abs(dr) + abs(dc))
                
                # Add fire proximity penalty
                safety_score += fire_count * 5
                
                # Add distance penalty (prefer shorter paths)
                safety_score += i * 0.1
            
            # If this is the best exit so far, update
            if safety_score < best_score:
                best_score = safety_score
                best_exit = exit_pos
                best_path = path
    
    # If no safe path found, try allowing fire cells
    if best_exit is None:
        for exit_pos in exits:
            path = astar_heat(
                grid, 
                robot, 
                exit_pos, 
                heatmap=heatmap,
                allow_fire=True,  # Allow going through fire
                fire_map=fire_map
            )
            if path:
                return exit_pos, path
    
    return best_exit, best_path

class Planner:
    def __init__(self):
        self.last_heatmap = None
        self.last_fire_map = None
    
    def plan(self, grid, robot, exits, heatmap):
        """Plan a path to the best exit."""
        if not exits:
            return None
            
        # Update fire map for faster access
        fire_map = (grid.arr == 2)  # Boolean array of fire cells
        
        # Choose best exit with path
        best_exit, path = choose_best_exit(robot, exits, grid, None, heatmap, fire_map)
        
        if path:
            return path
            
        # If no path found, try to move away from fire
        return self.emergency_evade(grid, robot, fire_map)
    
    def emergency_evade(self, grid, robot, fire_map):
        """Emergency movement away from fire when no path is found."""
        r, c = robot
        best_dir = None
        min_fire = float('inf')
        
        # Check all 8 directions
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid.h and 0 <= nc < grid.w and 
                grid.arr[nr, nc] != 1):  # Not a wall
                # Count fire in 3x3 area
                fire_count = 0
                for r2 in range(max(0, nr-1), min(grid.h, nr+2)):
                    for c2 in range(max(0, nc-1), min(grid.w, nc+2)):
                        if fire_map[r2, c2]:
                            fire_count += 1
                
                if fire_count < min_fire:
                    min_fire = fire_count
                    best_dir = (nr, nc)
        
        return [best_dir] if best_dir is not None else None
