import heapq
import numpy as np

def astar_heat(grid, start, goal, heatmap=None, allow_fire=False):
    """Robust A* pathfinding with improved path reconstruction and dynamic obstacle handling.
    
    Args:
        grid: Grid object with arr attribute (0=empty, 1=wall, 2=fire)
        start: Starting position (row, col)
        goal: Goal position (row, col)
        heatmap: Optional 2D array of heat values from fire
        allow_fire: If True, allows pathing through fire cells with high cost
    
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

    # Pre-compute all possible neighbors for each cell (4-way movement)
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.h and 0 <= nc < grid.w:
                neighbors.append((nr, nc))
        return neighbors
    
    # Heuristic function: Manhattan distance with tie-breaker
    def heuristic(a, b):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return (dx + dy) * 1.0001  # Slight tie-breaker
    
    # Check if position is walkable
    def is_walkable(pos):
        r, c = pos
        if not (0 <= r < grid.h and 0 <= c < grid.w):
            return False
        if grid.arr[r, c] == 1:  # Wall
            return False
        if grid.arr[r, c] == 2 and not allow_fire:  # Fire
            return False
        return True
    
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
        
        # Explore neighbors
        for neighbor in get_neighbors(current):
            if not is_walkable(neighbor):
                continue
                
            # Calculate movement cost
            if grid.arr[neighbor] == 2:  # Fire cell
                move_cost = 50.0  # High cost for moving through fire
            elif heatmap is not None and 0 <= neighbor[0] < heatmap.shape[0] and 0 <= neighbor[1] < heatmap.shape[1]:
                move_cost = 1.0 + (heatmap[neighbor[0], neighbor[1]] * 0.1)  # Heat influence
            else:
                move_cost = 1.0  # Default cost
            
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
        
        # If no exit-adjacent cells, find the position with lowest heat or closest to goal
        if heatmap is not None and heatmap.size > 0:
            def get_heat(p):
                if 0 <= p[0] < heatmap.shape[0] and 0 <= p[1] < heatmap.shape[1]:
                    return heatmap[p[0], p[1]]
                return float('inf')
                
            best_pos = min(came_from.keys(),
                         key=lambda p: (get_heat(p), abs(p[0]-goal[0]) + abs(p[1]-goal[1])))
        else:
            # Fallback: find position closest to goal
            best_pos = min(came_from.keys(),
                         key=lambda p: (abs(p[0]-goal[0]) + abs(p[1]-goal[1])))
        
        # Reconstruct path to this position
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
    
    # If absolutely no path was found, try to move to any adjacent cell
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if (0 <= nr < grid.h and 0 <= nc < grid.w and 
            grid.arr[nr, nc] != 1):  # Not a wall
            return [(nr, nc)]
    
    return None  # No possible moves
