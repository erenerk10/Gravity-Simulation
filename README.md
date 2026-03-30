# Gravity Simulation
 
A 2D gravity simulation built with Python and Pygame. Particles interact via Newtonian gravity, collide and merge, and can collapse into black holes that devour everything nearby.
  
## Features
 
- N-body gravity simulation using NumPy vectorization
- Velocity Verlet integration for stable orbits
- Particle merging with momentum conservation
- Black hole formation and event horizon absorption
- Real-time control panel (G, black hole mass threshold, time scale)
- Galaxy orbit system with a single keypress
 
---
 
## Installation
 
Make sure you have Python 3.8 or higher installed, then install the dependencies:
 
```bash
pip install -r requirements.txt
```
 
Then run the simulation:
 
```bash
python simulation.py
```
 
---
 
## Controls
 
| Input | Action |
|---|---|
| Left click | Spawn a particle |
| Right click | Spawn a black hole |
| `G` | Generate a galaxy orbit system |
| `Space` | Burst spawn 30 random particles |
| `C` | Clear all particles |
 
The panel in the bottom-left corner lets you adjust simulation parameters in real time:
 
| Slider | Description |
|---|---|
| G (Gravity) | Gravitational constant — higher values = stronger pull |
| BH Mass | Mass threshold at which a merged particle becomes a black hole |
| Time Scale | Simulation speed multiplier |
 
---
 
## How It Works
 
### Gravity
Every particle exerts a gravitational pull on every other particle according to Newton's law of universal gravitation. To prevent forces from becoming infinitely large when two particles are very close, a softening factor is added to the distance calculation.
 
### Integration
Particle positions and velocities are updated using the Velocity Verlet method, which conserves energy much better than basic Euler integration. This allows particles to maintain stable orbits instead of spiraling inward.
 
### Collisions
When two particles overlap, they merge. The resulting particle inherits the combined mass and the momentum of both, conserving linear momentum. If the merged mass exceeds the black hole threshold, it becomes a black hole.
 
### Black Holes
Black holes exert a stronger gravitational pull than normal particles. Any particle that crosses the event horizon (determined by the Schwarzschild multiplier) is absorbed and removed from the simulation.
 
### Performance

The simulation originally used NumPy vectorized matrix operations for gravity calculations, which improved performance compared to nested Python loops but still had **O(n²)** time complexity.

The current version uses the **Barnes–Hut algorithm** with a quadtree spatial partitioning system, reducing the complexity to approximately **O(n log n)**.
This allows the simulation to handle significantly larger particle counts while maintaining real-time performance.
 
---
 
## Requirements
 
- Python 3.8+
- pygame
- pygame_gui
- numpy
 
---
 
## TODO
 
- [x] Barnes-Hut algorithm for better performance at very high particle counts
- [ ] Gravitational wave visualization on black hole mergers
- [ ] Save / load simulation state
- [ ] Adjustable screen resolution
- [x] Particle color based on velocity
