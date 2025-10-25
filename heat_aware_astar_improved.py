import heapq
import numpy as np

def astar_heat(grid, start, goal, heatmap=None, allow_fire=False, fire_map=None):
    """Improved A* pathfinding with better fire avoidance and escape strategies.
    
    Args:
        grid: Grid object with arr attribute (0=empty, 1=wall, 2=fire)
        start: Starting position (row, col)
        goal: Goal position (row, col)
        heatmap: 2D array of heat values from fire (optional)
        allow_fire: If True, allows pathing through fire cells with high cost
        fire_map: Direct fire map for faster access (grid.arr == C.FIRE)
    
    Returns:
        List of (row, col) positions representing the path from start to goal,
        or None if no valid path exists.
    """
    # Input validation
    if not hasattr(grid, 'arr') or not hasattr(grid, 'h') or not hasattr(grid, 'w'):
        raise ValueError("Invalid grid object. Must have 'arr', 'h' and 'w' attributes")
    
    if not (0 <= start[0] < grid.h and 0 <= start[1] < grid.w):
        raise ValueError(f"Start position {start} is out of grid bounds")
    
    if not (0 <= goal[0] < grid.h and 0 <= goal[1] < grid.w):
        raise ValueError(f"Goal position {goal} is out of grid bounds")
    
    if heatmap is not None and (heatmap.shape[0] != grid.h or heatmap.shape[1] != grid.w):
        raise ValueError("Heatmap dimensions must match grid dimensions")
    
    # If start is the goal, return empty path
    if start == goal:
        return []

    # Pre-compute all possible neighbors for each cell (8-way movement for more options)
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        # Check 8 directions for more path options
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.h and 0 <= nc < grid.w:
                neighbors.append((nr, nc))
        return neighbors
    
    # Improved heuristic function with better tie-breaking
    def heuristic(a, b):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        # Use diagonal distance for better 8-way movement
        return max(dx, dy) + (2**0.5 - 1) * min(dx, dy)
    
    # Enhanced walkability check with fire awareness
    def is_walkable(pos, current_heat=0):
        r, c = pos
        if not (0 <= r < grid.h and 0 <= c < grid.w):
            return False, float('inf')
            
        # Wall
        if grid.arr[r, c] == 1:
            return False, float('inf')
            
        # Fire cell handling
        if grid.arr[r, c] == 2:
            if not allow_fire:
                return False, float('inf')
            # Allow fire cells but with very high cost
            return True, 50.0  # High cost for fire cells
            
        # Calculate heat cost if heatmap is available
        heat_cost = 0
        if heatmap is not None and 0 <= r < heatmap.shape[0] and 0 <= c < heatmap.shape[1]:
            heat_cost = heatmap[r, c] * 0.2  # Increased heat influence
            
        # Check for nearby fire (within 2 cells)
        fire_penalty = 0
        if fire_map is not None:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid.h and 0 <= nc < grid.w and 
                        fire_map[nr, nc] and (dr != 0 or dc != 0)):
                        # Add penalty based on distance to fire
                        dist = max(1, abs(dr) + abs(dc))
                        fire_penalty += 10.0 / (dist * dist)
        
        # Base movement cost + heat influence + fire proximity penalty
        total_cost = 1.0 + heat_cost + fire_penalty
        
        return True, total_cost
    
    # Check if we've reached the goal or adjacent to it
    def is_goal_reached(pos):
        return (abs(pos[0] - goal[0]) <= 1 and abs(pos[1] - goal[1]) <= 1)
    
    # Initialize data structures
    open_set = []
    heapq.heappush(open_set, (0, 0, start))  # (f_score, counter, position)
    
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    
    open_set_hash = {start}  # For O(1) lookups
    counter = 1  # For stable sorting in heap
    
    # Cache for walkable checks to avoid redundant calculations
    walkable_cache = {}
    
    while open_set:
        # Get the node with lowest f_score
        current_f, _, current = heapq.heappop(open_set)
        
        # Check if we've already processed this node with a better path
        if current not in open_set_hash:
            continue
            
        open_set_hash.remove(current)
        
        # Check if we've reached the goal
        if is_goal_reached(current):
            # Reconstruct path
            path = []
            if current != goal:
                path.append(goal)
            while current != start:
                path.append(current)
                current = came_from.get(current, start)
                if current is None:  # Shouldn't happen, but just in case
                    break
            path.reverse()
            return path
        
        # Get walkable neighbors with their costs
        for neighbor in get_neighbors(current):
            if neighbor in walkable_cache:
                is_walk, move_cost = walkable_cache[neighbor]
            else:
                is_walk, move_cost = is_walkable(neighbor, g_score.get(current, 0))
                walkable_cache[neighbor] = (is_walk, move_cost)
                
            if not is_walk:
                continue
            
            # Calculate tentative g score
            tentative_g_score = g_score[current] + move_cost
            
            # If this is a better path to the neighbor
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = tentative_g_score + heuristic(neighbor, goal)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f, counter, neighbor))
                    open_set_hash.add(neighbor)
                    counter += 1
    
    # If we get here, no direct path was found to the goal
    # Try to find the best possible path to any exit
    if came_from:
        # Get all explored cells that are adjacent to an exit
        exit_adjacent = []
        for pos in came_from:
            if is_goal_reached(pos):
                exit_adjacent.append(pos)
        
        # If we found cells adjacent to exits, return the best one
        if exit_adjacent:
            best_pos = min(exit_adjacent, key=lambda p: g_score.get(p, float('inf')))
            path = []
            current = best_pos
            while current != start and current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path if path else None
        
        # If no exit-adjacent cells, find the safest position
        def get_safety_score(p):
            # Prioritize positions with lower heat and closer to goal
            heat = heatmap[p[0], p[1]] if heatmap is not None and 0 <= p[0] < heatmap.shape[0] and 0 <= p[1] < heatmap.shape[1] else 0
            dist_to_goal = abs(p[0]-goal[0]) + abs(p[1]-goal[1])
            return heat * 0.7 + dist_to_goal * 0.3
            
        if came_from:
            best_pos = min(came_from.keys(), key=get_safety_score)
            path = []
            current = best_pos
            while current != start and current in came_from:
                path.append(current)
                current = came_from[current]
                # Prevent potential infinite loops
                if len(path) > len(came_from) * 2:
                    break
            path.reverse()
            return path if path else None
    
    # If absolutely no path was found, try to move to any adjacent cell away from fire
    if fire_map is not None:
        # Find direction with least fire
        best_dir = None
        min_fire = float('inf')
        
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-way movement
            nr, nc = start[0] + dr, start[1] + dc
            if (0 <= nr < grid.h and 0 <= nc < grid.w and 
                grid.arr[nr, nc] != 1):  # Not a wall
                # Count fire in 3x3 area
                fire_count = 0
                for r in range(max(0, nr-1), min(grid.h, nr+2)):
                    for c in range(max(0, nc-1), min(grid.w, nc+2)):
                        if fire_map[r, c]:
                            fire_count += 1
                
                if fire_count < min_fire:
                    min_fire = fire_count
                    best_dir = (nr, nc)
        
        if best_dir is not None:
            return [best_dir]
    
    # Last resort: try any adjacent cell that's not a wall
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-way movement
        nr, nc = start[0] + dr, start[1] + dc
        if (0 <= nr < grid.h and 0 <= nc < grid.w and 
            grid.arr[nr, nc] != 1):  # Not a wall
            return [(nr, nc)]
    
    return None  # No possible moves
