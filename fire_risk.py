import numpy as np

class FireRiskAnalyzer:
    def __init__(self, risk_decay=0.7):
        self.risk_decay = risk_decay
        self.risk_map = None
        self.previous_fire_map = None
    
    def update_risk(self, fire_map):
        """Update risk map based on current fire positions."""
        if self.risk_map is None or self.risk_map.shape != fire_map.shape:
            self.risk_map = np.zeros_like(fire_map, dtype=float)
        
        # Apply decay to existing risks
        self.risk_map *= self.risk_decay
        
        # Add new risks from current fire positions
        new_risks = np.zeros_like(fire_map, dtype=float)
        
        # Find all fire cells
        fire_cells = np.argwhere(fire_map)
        
        # Calculate risk propagation from each fire cell
        for (r, c) in fire_cells:
            # Immediate cell gets highest risk
            new_risks[r, c] = max(new_risks[r, c], 1.0)
            
            # Spread risk to adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < fire_map.shape[0] and 
                        0 <= nc < fire_map.shape[1] and 
                        not fire_map[nr, nc]):  # Don't overwrite actual fire
                        distance = max(abs(dr), abs(dc))  # Chebyshev distance
                        risk = 0.7 / (distance + 1)  # Decrease with distance
                        new_risks[nr, nc] = max(new_risks[nr, nc], risk)
        
        # Update risk map with new risks
        self.risk_map = np.maximum(self.risk_map, new_risks)
        self.previous_fire_map = fire_map.copy()
        
        return self.risk_map
    
    def get_risk_level(self, pos):
        """Get risk level at position (r, c)"""
        if self.risk_map is None:
            return 0.0
        r, c = int(pos[0]), int(pos[1])
        if 0 <= r < self.risk_map.shape[0] and 0 <= c < self.risk_map.shape[1]:
            return self.risk_map[r, c]
        return 0.0
    
    def get_risk_category(self, pos):
        """Get risk category (low/medium/high) for a position"""
        risk = self.get_risk_level(pos)
        if risk < 0.3:
            return 'low'
        elif risk < 0.7:
            return 'medium'
        else:
            return 'high'
