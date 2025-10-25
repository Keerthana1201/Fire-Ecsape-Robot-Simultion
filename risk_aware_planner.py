from heat_aware_astar_improved import astar_heat
from fire_risk import FireRiskAnalyzer
import numpy as np

class RiskAwarePlanner:
    def __init__(self):
        self.risk_analyzer = FireRiskAnalyzer()
        self.last_heatmap = None
        self.last_fire_map = None
    
    def plan(self, grid, robot, exits, heatmap):
        """Plan a path considering fire risks."""
        if not exits:
            return None
            
        # Update fire map and risk analysis
        fire_map = (grid.arr == 2)  # Boolean array of fire cells
        risk_map = self.risk_analyzer.update_risk(fire_map)
        
        # Combine heatmap and risk map
        if heatmap is not None:
            # Normalize heatmap to 0-1 range if needed
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Combine heat and risk (weighted sum)
            combined_risk = 0.6 * risk_map + 0.4 * (heatmap if heatmap is not None else 0)
        else:
            combined_risk = risk_map
        
        # Adjust path planning based on risk levels
        def risk_aware_cost(pos):
            r, c = int(pos[0]), int(pos[1])
            if not (0 <= r < combined_risk.shape[0] and 0 <= c < combined_risk.shape[1]):
                return float('inf')
                
            risk = combined_risk[r, c]
            
            # Base cost increases with risk
            cost = 1.0 + (risk * 5.0)  # Scale risk to have meaningful impact
            
            # Additional cost for actual fire cells
            if fire_map[r, c]:
                cost += 20.0  # Very high cost for actual fire
                
            return cost
        
        # Find best exit considering risk
        best_path = None
        best_score = float('inf')
        
        for exit_pos in exits:
            # Try to find path with risk awareness
            path = astar_heat(
                grid=grid,
                start=robot,
                goal=exit_pos,
                heatmap=combined_risk,
                allow_fire=False,
                fire_map=fire_map
            )
            
            if path:
                # Calculate path score based on risk
                path_risk = sum(risk_aware_cost(p) for p in path)
                path_length = len(path)
                
                # Combine risk and length (prefer shorter, safer paths)
                score = path_risk * (1 + 0.1 * path_length)  # Length penalty
                
                if score < best_score:
                    best_score = score
                    best_path = path
        
        # If no safe path found, try allowing higher risk paths
        if best_path is None:
            for exit_pos in exits:
                path = astar_heat(
                    grid=grid,
                    start=robot,
                    goal=exit_pos,
                    heatmap=combined_risk * 0.5,  # Reduce risk impact
                    allow_fire=True,  # Allow going through fire if needed
                    fire_map=fire_map
                )
                if path:
                    return path
        
        # If still no path, use emergency evasion
        if best_path is None:
            return self.emergency_evade(grid, robot, fire_map, combined_risk)
            
        return best_path
    
    def emergency_evade(self, grid, robot, fire_map, risk_map):
        """Emergency movement away from fire when no clear path exists."""
        r, c = robot
        best_dir = None
        best_score = -float('inf')
        
        # Check all 8 directions
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid.h and 0 <= nc < grid.w and 
                grid.arr[nr, nc] != 1):  # Not a wall
                
                # Calculate safety score (higher is better)
                safety = 0
                
                # Avoid fire cells
                if fire_map[nr, nc]:
                    continue
                
                # Prefer lower risk areas
                risk = risk_map[nr, nc] if nr < risk_map.shape[0] and nc < risk_map.shape[1] else 0
                safety += (1 - risk) * 10
                
                # Prefer areas with fewer nearby fire cells
                fire_count = 0
                for r2 in range(max(0, nr-2), min(grid.h, nr+3)):
                    for c2 in range(max(0, nc-2), min(grid.w, nc+3)):
                        if fire_map[r2, c2]:
                            distance = max(abs(nr - r2), abs(nc - c2))
                            fire_count += 1.0 / (distance + 1)
                
                safety -= fire_count * 2
                
                # Prefer moving toward exits if possible
                if hasattr(grid, 'exits'):
                    min_exit_dist = min(
                        (abs(nr - e[0]) + abs(nc - e[1]) for e in grid.exits),
                        default=0
                    )
                    safety += 5.0 / (min_exit_dist + 1)
                
                if safety > best_score:
                    best_score = safety
                    best_dir = (nr, nc)
        
        return [best_dir] if best_dir is not None else None
    
    def get_risk_category(self, pos):
        """Get risk category for visualization."""
        return self.risk_analyzer.get_risk_category(pos)
