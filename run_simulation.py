import sys
import os
from src.gui_risk_aware import RiskAwareApp
import tkinter as tk

def main():
    # Create the main window
    root = tk.Tk()
    root.title("🚒 Fire Escape Robot Simulation")
    root.geometry("1000x800")
    
    # Default map path - adjust if needed
    map_path = os.path.join("maps", "default.txt")
    
    # Create the app
    app = RiskAwareApp(root, map_path)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
