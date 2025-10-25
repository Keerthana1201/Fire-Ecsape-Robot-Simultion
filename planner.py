try:
    from .heat_aware_astar import astar_heat
except ImportError:
    from heat_aware_astar import astar_heat
import random

def choose_exit(robot, exits):
    return min(exits, key=lambda e: abs(robot[0]-e[0])+abs(robot[1]-e[1]))

def choose_safest_exit(robot, exits, grid, fire, allow_fire=False):
    """Choose the exit that has the safest path considering current fire state"""
    best_exit = None
    best_score = float('inf')
    best_path = None
    
    for exit_pos in exits:
        # Calculate path safety score
        path = astar_heat(grid, robot, exit_pos, fire.compute_heatmap(grid), allow_fire=allow_fire)
        if path:
            # Calculate safety score (lower is better)
            safety_score = 0
            fire_cells_in_path = 0
            for i, (r, c) in enumerate(path):
                # Check if this cell will be on fire soon
                if grid.arr[r, c] == 2:  # Already on fire
                    safety_score += 1000
                    fire_cells_in_path += 1
                else:
                    # Check surrounding fire density
                    fire_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if grid.in_bounds(nr, nc) and grid.arr[nr, nc] == 2:
                                fire_count += 1
                    safety_score += fire_count * 10 + i  # Distance penalty
                    
            if safety_score < best_score:
                best_score = safety_score
                best_exit = exit_pos
                best_path = path
    
    return best_exit if best_exit else choose_exit(robot, exits)

class Planner:
    def __init__(self):
        pass
    
    def plan(self, grid, robot, exits, heatmap):
        goal = choose_exit(robot, exits)
        return astar_heat(grid, robot, goal, heatmap)
    
    def plan_with_fire_avoidance(self, grid, robot, exits, heatmap, fire, max_attempts=3):
        """Plan path with dynamic fire avoidance and emergency escape
        
        Args:
            grid: The grid/map
            robot: Current robot position (row, col)
            exits: List of exit positions [(row, col), ...]
            heatmap: Heatmap of fire spread
            fire: Fire simulation object
            max_attempts: Maximum number of pathfinding attempts
        """
        if not exits:
            return None
            
        # Try to find a safe path to any exit
        for attempt in range(max_attempts):
            # Alternate between different exit selection strategies
            if attempt == 0:
                # First try: Safest path to closest exit
                goal = min(exits, key=lambda e: abs(robot[0]-e[0]) + abs(robot[1]-e[1]))
                allow_fire = False
            elif attempt == 1:
                # Second try: Safest path considering fire
                goal = choose_safest_exit(robot, exits, grid, fire, allow_fire=False)
                allow_fire = False
            else:
                # Final try: Allow going through fire if needed
                goal = choose_safest_exit(robot, exits, grid, fire, allow_fire=True)
                allow_fire = True
                
            path = astar_heat(grid, robot, goal, heatmap, allow_fire=allow_fire)
            
            if path and len(path) > 1:
                # Check if the next move is safe
                next_cell = path[1]
                if self.is_cell_safe(next_cell, grid, fire) or attempt == max_attempts - 1:
                    if allow_fire:
                        print(f"✓ Emergency path found to exit {goal}")
                    return path
        
        # If we get here, no safe path was found after all attempts
        print("⚠️ No safe path found! Using emergency evasion...")
        return self.find_emergency_escape(grid, robot, exits, heatmap, fire)
    
    def is_cell_safe(self, cell, grid, fire):
        """Check if a cell is safe to move to"""
        r, c = cell
        if not grid.in_bounds(r, c):
            return False
            
        # Cell is not safe if it's a wall
        if grid.arr[r, c] == 1:
            return False
            
        # Cell is not safe if it's on fire
        if grid.arr[r, c] == 2:  # FIRE
            return False
            
        # Cell is not safe if it's surrounded by too much fire
        fire_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if grid.in_bounds(nr, nc) and grid.arr[nr, nc] == 2:
                    fire_count += 1
                    
        # If more than 4 neighbors are on fire, it's too dangerous
        return fire_count <= 4
    
    def find_alternative_path(self, grid, robot, exits, heatmap, fire):
        """Find alternative path when current path is blocked by fire"""
        # Try each exit in order of distance
        sorted_exits = sorted(exits, key=lambda e: abs(robot[0]-e[0])+abs(robot[1]-e[1]))
        
        for exit_pos in sorted_exits:
            # Try safe path first
            path = astar_heat(grid, robot, exit_pos, heatmap, allow_fire=False)
            if path and len(path) > 1:
                next_cell = path[1]
                if self.is_cell_safe(next_cell, grid, fire):
                    print(f"✓ Switching to alternative exit: {exit_pos}")
                    return path
        
        # If no safe path found, try emergency mode (through fire)
        print("⚠️ No safe alternative found! Trying emergency paths...")
        for exit_pos in sorted_exits:
            path = astar_heat(grid, robot, exit_pos, heatmap, allow_fire=True)
            if path and len(path) > 1:
                exit_clean = (int(exit_pos[0]), int(exit_pos[1]))
                print(f"✓ Emergency path found to exit: {exit_clean}")
                return path
        
        # Last resort: move to any safer adjacent cell
        return self.find_emergency_escape(grid, robot, exits, heatmap, fire)
    
    def find_emergency_escape(self, grid, robot, exits, heatmap, fire):
        """Find emergency escape when all paths are blocked"""
        r, c = robot
        
        # First try: Find safe moves (not on fire, not too dangerous)
        safe_moves = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (grid.in_bounds(nr, nc) and 
                grid.arr[nr, nc] != 1 and  # Not a wall
                grid.arr[nr, nc] != 2 and  # Not on fire
                self.is_cell_safe((nr, nc), grid, fire)):
                safe_moves.append((nr, nc))
        
        if safe_moves:
            # Choose the move that gets us closer to nearest exit
            best_move = min(safe_moves, key=lambda move: 
                          min(abs(move[0] - exit[0]) + abs(move[1] - exit[1]) 
                              for exit in exits))
            print(f"✓ Emergency escape move to safe cell: {best_move}")
            return [robot, best_move]
        
        # Second try: Move to any non-wall cell (even if on fire - survival attempt)
        any_moves = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if grid.in_bounds(nr, nc) and grid.arr[nr, nc] != 1:  # Not a wall
                # Calculate danger level
                danger = 0
                if grid.arr[nr, nc] == 2:  # On fire
                    danger = 100
                else:
                    # Count surrounding fire
                    for dr2 in [-1, 0, 1]:
                        for dc2 in [-1, 0, 1]:
                            nr2, nc2 = nr + dr2, nc + dc2
                            if grid.in_bounds(nr2, nc2) and grid.arr[nr2, nc2] == 2:
                                danger += 1
                any_moves.append((nr, nc, danger))
        
        if any_moves:
            # Choose least dangerous move toward nearest exit
            best_move = min(any_moves, key=lambda m: 
                          (m[2], min(abs(m[0] - exit[0]) + abs(m[1] - exit[1]) for exit in exits)))
            print(f"⚠️ CRITICAL: Moving through danger to: {best_move[:2]}")
            return [robot, best_move[:2]]
        
        # Absolute last resort: stay put
        print("❌ TRAPPED: No moves available, staying in place")
        return [robot]
