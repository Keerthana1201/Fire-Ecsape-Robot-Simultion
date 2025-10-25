import numpy as np
import random
try:
    from . import cell_types as C
except ImportError:
    import cell_types as C

class FireSpread:
    def __init__(self, prob=0.25, seed=None):
        self.prob = prob
        self.base_prob = prob  # Store original probability
        self.rng = random.Random(seed)

    def step(self, grid, excludes=(), robot_pos=None):
        """Spread fire with optional robot position awareness
        
        Args:
            grid: Grid object
            excludes: Positions to exclude from fire spread (e.g., exits)
            robot_pos: Robot position (r, c) - if provided, fire spread slows near robot
        """
        arr = grid.arr
        h,w = arr.shape
        to_burn = []
        
        # Check if robot is in danger (surrounded by fire)
        robot_danger = False
        if robot_pos:
            fire_count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = robot_pos[0] + dr, robot_pos[1] + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == C.FIRE:
                        fire_count += 1
            robot_danger = fire_count >= 4  # Robot is in danger if 4+ neighbors on fire
        
        # Adjust spread probability if robot is in danger
        current_prob = self.prob * 0.5 if robot_danger else self.prob
        
        for i in range(h):
            for j in range(w):
                if arr[i,j] == C.FIRE:
                    # Spread to 4-directional neighbors
                    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0<=ni<h and 0<=nj<w:
                            if arr[ni,nj] in (C.EMPTY, C.START) and (ni,nj) not in excludes:
                                # Further reduce spread near robot
                                spread_prob = current_prob
                                if robot_pos and abs(ni - robot_pos[0]) + abs(nj - robot_pos[1]) <= 2:
                                    spread_prob *= 0.7  # 30% reduction near robot
                                
                                if self.rng.random() < spread_prob:
                                    to_burn.append((ni,nj))
                    
                    # Occasionally spread to diagonal neighbors to create more complex fire patterns
                    if self.rng.random() < 0.3:  # 30% chance for diagonal spread - creates complex challenges
                        for di,dj in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            ni, nj = i+di, j+dj
                            if 0<=ni<h and 0<=nj<w:
                                if arr[ni,nj] in (C.EMPTY, C.START) and (ni,nj) not in excludes:
                                    spread_prob = current_prob * 0.5
                                    if robot_pos and abs(ni - robot_pos[0]) + abs(nj - robot_pos[1]) <= 2:
                                        spread_prob *= 0.7
                                    
                                    if self.rng.random() < spread_prob:
                                        to_burn.append((ni,nj))
        
        for (i,j) in to_burn:
            arr[i,j] = C.FIRE

    def compute_heatmap(self, grid):
        h,w = grid.arr.shape
        heat = np.zeros((h,w), dtype=float)
        fire_cells = list(zip(*((grid.arr==C.FIRE).nonzero())))
        for (fi,fj) in fire_cells:
            for i in range(h):
                for j in range(w):
                    d = abs(fi-i)+abs(fj-j)
                    if d==0:
                        heat[i,j] += 100.0
                    else:
                        heat[i,j] += max(0.0, 5.0/d)
        return heat
