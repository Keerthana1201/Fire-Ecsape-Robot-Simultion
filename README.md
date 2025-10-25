# Fire-Ecsape-Robot-Simultion

​This project simulates an intelligent Fire Escape Robot Path Planner designed to assist in finding safe and efficient evacuation routes in burning buildings. It addresses the challenge of static evacuation plans failing in dynamic environments where fire spreads and exits become blocked.  

# Key Features & Objectives

​The primary goal is to design a simulation that models a fire emergency and adapts the robot's escape route dynamically.  
​2D Grid Modeling: The building is modeled as a 2D grid comprising cells for open space, walls, fire zones, and exit points.  
​A* Pathfinding: The project applies the A^* algorithm to compute the shortest and safest paths, treating fire zones as blocked or high-cost cells.  
​Dynamic Fire Simulation: The simulation incorporates step-by-step fire spread, prompting the pathfinding algorithm to recalculate routes dynamically when conditions change.  
​Visualization: Interactive, animated visualization of the robot's movement, the fire spread, and the chosen safe path using Pygame/Tkinter.  
​Evaluation: Performance is evaluated based on path length, safety, and the system's adaptability to different fire spread scenarios.  
​
# Methodology Overview

​The simulation is built around a six-step process to ensure dynamic and safe path planning.  
​Environment Setup: Model the building as a 2D grid using NumPy arrays for efficient representation.  
​Robot Initialization: Define the robot's starting position and one or more exit points.  
​Pathfinding Algorithm: Execute the A^* algorithm for safe route computation, utilizing priority queues (heapq) for efficiency.  
​Dynamic Fire Simulation: Introduce fire spread over time and ensure the path is re-evaluated immediately upon environmental change.  
​Visualization & Simulation: Use Matplotlib for static grid display and Pygame/Tkinter for interactive, animated robot movement.  

# ​Output & Evaluation: Display the final escape path and evaluate the performance against various scenarios.
