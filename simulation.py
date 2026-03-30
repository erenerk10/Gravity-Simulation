import pygame
import math
import random
import pygame_gui

pygame.init()

# --- Screen settings ---
width, height = 1920, 1080  # Resolution: Full HD
screen = pygame.display.set_mode((width, height))  # Create the display window
pygame.display.set_caption('Gravity Simulation — Barnes-Hut')  # Window title

clock = pygame.time.Clock()  # Clock object to control frame rate
FPS = 60  # Target frames per second

# --- Simulation parameters ---
G = 0.5                         # Gravitational constant
softening = 10                  # Softening factor to prevent singularities at very short distances
black_hole_mass = 500           # Mass threshold at which a particle becomes a black hole
time_scale = 1.0                # Simulation speed multiplier
SCHWARZSCHILD_MULTIPLIER = 3    # Event horizon radius = particle radius * this value
THETA = 1.25                    # Barnes-Hut angle criterion. Smaller = more accurate but slower, larger = faster but rougher.
MAX_SPEED = 500                 # Speed cap to prevent particles from moving too fast
COLOR_MAX_SPEED = 60.0          # Reference speed used for color normalization (smaller -> pushes colors toward white/yellow/red sooner)
COLOR_DECAY = 0.96              # Per-frame decay multiplier applied to the running color maximum
running_color_max = COLOR_MAX_SPEED  # Adaptive maximum speed used for color scaling

# Performance settings
MAX_TRAIL_LENGTH = 40           # Maximum number of trail points stored per particle
COLLISION_GRID_SIZE = 80        # Cell size for the spatial hash grid used in collision detection

# Speed -> color map (dark blue -> light blue -> white -> yellow -> red)
def speed_to_color(norm_speed):
    norm = min(max(norm_speed, 0.0), 1.0)  # Clamp normalized speed to [0, 1]
    # Color keypoints along the gradient
    c0 = (0, 0, 128)       # Dark blue  (slowest)
    c1 = (0, 128, 255)     # Light blue
    c2 = (255, 255, 255)   # White
    c3 = (255, 255, 0)     # Yellow
    c4 = (255, 0, 0)       # Red        (fastest)

    # Linear interpolation between adjacent color stops
    if norm <= 0.25:
        t = norm / 0.25
        return (int(c0[0] + (c1[0] - c0[0])*t), int(c0[1] + (c1[1] - c0[1])*t), int(c0[2] + (c1[2] - c0[2])*t))
    if norm <= 0.5:
        t = (norm - 0.25) / 0.25
        return (int(c1[0] + (c2[0] - c1[0])*t), int(c1[1] + (c2[1] - c1[1])*t), int(c1[2] + (c2[2] - c1[2])*t))
    if norm <= 0.75:
        t = (norm - 0.5) / 0.25
        return (int(c2[0] + (c3[0] - c2[0])*t), int(c2[1] + (c3[1] - c2[1])*t), int(c2[2] + (c3[2] - c2[2])*t))
    t = (norm - 0.75) / 0.25
    return (int(c3[0] + (c4[0] - c3[0])*t), int(c3[1] + (c4[1] - c3[1])*t), int(c3[2] + (c4[2] - c3[2])*t))

particles = []  # Master list holding all active particles
font = pygame.font.SysFont(None, 20)  # Default font for HUD text

# --- GUI setup ---
manager = pygame_gui.UIManager((width, height))  # pygame_gui manager handles all UI elements

# Slider for gravitational constant G
g_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 900, 200, 30),
    start_value=G, value_range=(0.1, 5.0), manager=manager)

# Slider for the mass at which particles become black holes
bh_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 940, 200, 30),
    start_value=black_hole_mass, value_range=(100, 2000), manager=manager)

# Slider for simulation time scale
time_scale_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 980, 200, 30),
    start_value=time_scale, value_range=(0.1, 3.0), manager=manager)


# ---------------------------------------------------------------------------
# Barnes-Hut Quadtree
# ---------------------------------------------------------------------------

class QuadNode:
    """
    A single node in the Barnes-Hut quadtree.
    Each node represents a square region of 2D space and stores
    the combined center-of-mass and total mass of all particles inside it.
    """

    def __init__(self, cx, cy, half_size):
        self.cx = cx                # Center x of this node's bounding box
        self.cy = cy                # Center y of this node's bounding box
        self.half_size = half_size  # Half the side length of the bounding box

        self.total_mass = 0.0       # Sum of all particle masses in this node
        self.cm_x = 0.0            # Center-of-mass x coordinate
        self.cm_y = 0.0            # Center-of-mass y coordinate
        self.is_black_hole = False  # True if this node contains at least one black hole

        self.particle = None        # The single particle stored in a leaf node
        self.children = None        # Child nodes: [NW, NE, SW, SE]; None if this is a leaf

    def _subdivide(self):
        """Splits this node into four equal child quadrants."""
        q = self.half_size / 2      # Half-size of each child quadrant
        cx, cy = self.cx, self.cy
        self.children = [
            QuadNode(cx - q, cy - q, q),  # North-West
            QuadNode(cx + q, cy - q, q),  # North-East
            QuadNode(cx - q, cy + q, q),  # South-West
            QuadNode(cx + q, cy + q, q),  # South-East
        ]

    def _get_child(self, x, y):
        """Returns the child quadrant that contains the given coordinates."""
        east  = x >= self.cx   # Right half of the box
        south = y >= self.cy   # Bottom half of the box
        if not east and not south: return self.children[0]  # NW
        if east     and not south: return self.children[1]  # NE
        if not east and south:     return self.children[2]  # SW
        return self.children[3]                             # SE

    def insert(self, p):
        """
        Inserts a particle into the quadtree.
        - If the node is empty (leaf), store the particle here.
        - If the node is occupied (leaf), subdivide and push both particles into children.
        - If the node is internal, delegate to the appropriate child.
        """
        # Update center-of-mass and total mass incrementally
        if self.total_mass == 0:
            self.cm_x = p.x   # First particle: center-of-mass is just its position
            self.cm_y = p.y
        else:
            total = self.total_mass + p.mass
            self.cm_x = (self.cm_x * self.total_mass + p.x * p.mass) / total  # Weighted average x
            self.cm_y = (self.cm_y * self.total_mass + p.y * p.mass) / total  # Weighted average y

        self.total_mass += p.mass               # Accumulate total mass
        if p.is_black_hole:
            self.is_black_hole = True           # Propagate black-hole flag up the tree

        # Case 1: empty leaf node — just store the particle
        if self.particle is None and self.children is None:
            self.particle = p
            return

        # Case 2: occupied leaf node — subdivide and re-insert both particles
        if self.children is None:
            self._subdivide()
            old = self.particle          # The particle that was already here
            self.particle = None         # This node is no longer a leaf
            # Guard against identical positions causing infinite recursion
            if old.x != p.x or old.y != p.y:
                self._get_child(old.x, old.y).insert(old)  # Re-insert old particle
                self._get_child(p.x, p.y).insert(p)        # Insert new particle
            return

        # Case 3: internal node — pass directly to the correct child
        self._get_child(p.x, p.y).insert(p)

    def compute_force(self, p):
        """
        Computes the gravitational force this node exerts on particle p.
        Uses the Barnes-Hut approximation: if the node is far enough away
        (s/d < THETA), treat the entire node as a single point mass.

        Returns: (fx, fy) — force vector components
        """
        if self.total_mass == 0:
            return 0.0, 0.0  # Empty node contributes no force

        dx = self.cm_x - p.x   # Vector from particle to center-of-mass
        dy = self.cm_y - p.y
        dist_sq = dx * dx + dy * dy + softening ** 2  # Softened distance squared

        # Skip if this leaf node contains the particle itself
        if self.children is None and self.particle is p:
            return 0.0, 0.0

        # Guard against zero distance (should not normally occur after softening)
        if dist_sq <= 0.0:
            return 0.0, 0.0

        dist = math.sqrt(dist_sq)  # Actual (softened) distance

        # Barnes-Hut criterion: use approximation if the node is small relative to distance
        size = self.half_size * 2  # Full side length of this node's bounding box
        if self.children is None or (size * size) < (THETA * THETA) * dist_sq:
            # Apply extra gravity between black holes for dramatic interactions
            g_val = G * 5 if (p.is_black_hole or self.is_black_hole) else G
            force = g_val * p.mass * self.total_mass / dist_sq  # Newton's law (softened)
            fx = force * dx / dist   # Decompose into x component
            fy = force * dy / dist   # Decompose into y component
            return fx, fy

        # Criterion not met — recurse into children for a more accurate result
        fx, fy = 0.0, 0.0
        for child in self.children:
            cfx, cfy = child.compute_force(p)
            fx += cfx
            fy += cfy
        return fx, fy


# ---------------------------------------------------------------------------
# Particle class
# ---------------------------------------------------------------------------

class Particle:
    def __init__(self, x, y, mass):
        self.x = x                              # Current x position
        self.y = y                              # Current y position
        self.mass = mass                        # Particle mass
        self.radius = max(3, int(math.sqrt(mass)))  # Visual radius scales with mass
        self.vx = 0.0                           # Velocity x component
        self.vy = 0.0                           # Velocity y component
        self.ax = 0.0                           # Acceleration x component (updated each frame)
        self.ay = 0.0                           # Acceleration y component (updated each frame)
        self.trail = []                         # List of past positions for trail rendering
        self.max_trail = MAX_TRAIL_LENGTH       # Maximum number of trail points
        self.is_black_hole = False              # Whether this particle is a black hole
        self.color = (255, 255, 255)            # Render color (white by default)
        self.speed = 0.0                        # Scalar speed (magnitude of velocity)

    def update(self, dt):
        # Record current position into trail before moving
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)  # Remove oldest trail point to maintain max length

        # Symplectic Euler integration: update velocity then position
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        # Position updated using current velocity + half acceleration correction
        self.x  += self.vx * dt + 0.5 * self.ax * dt * dt
        self.y  += self.vy * dt + 0.5 * self.ay * dt * dt

        # Update scalar speed for color mapping
        self.speed = math.sqrt(self.vx**2 + self.vy**2)

    def draw(self, screen):
        # Draw fading trail: older points are dimmer
        if self.trail:
            for i, (tx, ty) in enumerate(self.trail):
                alpha = i / len(self.trail)            # 0.0 (oldest) → 1.0 (newest)
                r, g, b = self.color
                color = (int(r * alpha), int(g * alpha), int(b * alpha))  # Dim color by age
                pygame.draw.circle(screen, color, (int(tx), int(ty)), max(1, self.radius // 3))
        # Draw the particle itself at full color
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


# ---------------------------------------------------------------------------
# Barnes-Hut gravity solver (replaces the naive O(n²) apply_gravity)
# ---------------------------------------------------------------------------

def apply_gravity_bh(particles, dt):
    """
    Builds a fresh quadtree each frame and uses it to compute gravitational
    forces on all particles in O(n log n) time instead of O(n²).
    Also handles event-horizon absorption for black holes.
    """
    if len(particles) < 2:
        return  # Need at least two particles for any interaction

    # Compute a bounding box that encloses all particles with some margin
    margin = 100
    min_x = min(p.x for p in particles) - margin
    max_x = max(p.x for p in particles) + margin
    min_y = min(p.y for p in particles) - margin
    max_y = max(p.y for p in particles) + margin

    # Make the bounding box square and centered
    half_size = max(max_x - min_x, max_y - min_y) / 2
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # Build the quadtree root node
    root = QuadNode(cx, cy, half_size)

    # Insert all particles into the quadtree
    for p in particles:
        root.insert(p)

    # Compute force on each particle and convert to acceleration (F = ma → a = F/m)
    for p in particles:
        fx, fy = root.compute_force(p)
        p.ax = fx / p.mass
        p.ay = fy / p.mass

    # Event horizon absorption: remove particles that enter a black hole's event horizon
    to_remove = set()
    for bh in [p for p in particles if p.is_black_hole]:
        for p in [p for p in particles if not p.is_black_hole]:
            dx = bh.x - p.x
            dy = bh.y - p.y
            if math.sqrt(dx**2 + dy**2) < bh.radius * SCHWARZSCHILD_MULTIPLIER:
                to_remove.add(id(p))  # Mark particle for removal

    # Remove absorbed particles from the simulation
    particles[:] = [p for p in particles if id(p) not in to_remove]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def handle_collisions(particles):
    """
    Detects and resolves particle collisions using a spatial hash grid.
    When two particles overlap, they merge: the smaller is absorbed by the larger,
    conserving momentum. If the resulting mass exceeds the threshold, it becomes a black hole.
    """
    cell_size = COLLISION_GRID_SIZE
    grid = {}  # Maps grid cell (ix, iy) → list of particles in that cell

    # Assign each particle to its grid cell
    for p in particles:
        key = (int(p.x // cell_size), int(p.y // cell_size))
        grid.setdefault(key, []).append(p)

    to_remove = set()  # IDs of particles that have been absorbed this frame

    for cell, bucket in grid.items():
        for p1 in bucket:
            if id(p1) in to_remove:
                continue  # Skip already-absorbed particles

            # Check this cell and all 8 neighbouring cells
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_bucket = grid.get((cell[0] + dx, cell[1] + dy), [])
                    for p2 in neighbor_bucket:
                        if p1 is p2 or id(p2) in to_remove:
                            continue  # Skip self-comparison and already-absorbed particles

                        # Check for overlap using squared distances (avoids sqrt)
                        dx_ = p2.x - p1.x
                        dy_ = p2.y - p1.y
                        dist_sq = dx_ * dx_ + dy_ * dy_
                        radius_sum = p1.radius + p2.radius
                        if dist_sq >= radius_sum * radius_sum:
                            continue  # No collision

                        # Determine which particle survives (the more massive one)
                        big = p1 if p1.mass >= p2.mass else p2
                        small = p2 if p1.mass >= p2.mass else p1

                        if id(big) in to_remove or id(small) in to_remove:
                            continue  # One of them was already consumed this frame

                        # Inelastic merge: conserve momentum, combine mass
                        big.vx = (big.mass * big.vx + small.mass * small.vx) / (big.mass + small.mass)
                        big.vy = (big.mass * big.vy + small.mass * small.vy) / (big.mass + small.mass)
                        big.mass += small.mass
                        big.radius = max(3, int(math.sqrt(big.mass) * 2))  # Recalculate radius
                        # Promote to black hole if mass threshold is reached
                        if big.mass >= black_hole_mass and not big.is_black_hole:
                            big.is_black_hole = True
                            big.color = (180, 0, 255)  # Purple color for black holes

                        to_remove.add(id(small))  # Mark the smaller particle for removal

    # Remove all absorbed particles in one pass
    if to_remove:
        particles[:] = [p for p in particles if id(p) not in to_remove]


def draw_black_hole(screen, p):
    """Renders a black hole with a dark center and glowing purple accretion rings."""
    cx, cy = int(p.x), int(p.y)
    event_horizon = p.radius * SCHWARZSCHILD_MULTIPLIER  # Radius of the event horizon

    # Draw glowing rings from outermost to innermost (painter's algorithm)
    for ring in range(5, 0, -1):
        ring_radius = int(event_horizon + ring * 6)   # Each ring is offset by 6 pixels
        alpha_surf = pygame.Surface((ring_radius * 2, ring_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(alpha_surf, (180, 0, 255, ring * 12), (ring_radius, ring_radius), ring_radius)
        screen.blit(alpha_surf, (cx - ring_radius, cy - ring_radius))

    # Draw the dark event horizon (black filled circle)
    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), int(event_horizon))
    # Draw a bright purple outline at the event horizon boundary
    pygame.draw.circle(screen, (180, 0, 255), (cx, cy), int(event_horizon), 2)
    # Draw an outer glow ring just outside the event horizon
    pygame.draw.circle(screen, (220, 100, 255), (cx, cy), int(event_horizon) + 2, 1)


def create_orbit_system(cx, cy):
    """
    Creates a central black hole surrounded by 200 particles in stable circular orbits.
    Each particle's orbital velocity is computed from v = sqrt(G * M / r).
    """
    # Create the central massive black hole
    sun = Particle(cx, cy, 2000)
    sun.is_black_hole = True
    sun.color = (180, 0, 255)  # Purple
    particles.append(sun)

    # Spawn orbiting particles at random radii and angles
    for _ in range(200):
        r     = random.randint(120, 450)              # Orbital radius (pixels)
        angle = random.uniform(0, 2 * math.pi)        # Random angle around the center
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        p = Particle(x, y, random.uniform(5, 20))     # Small random mass
        v = math.sqrt(G * sun.mass / r)               # Circular orbit speed
        p.vx = -v * math.sin(angle)                   # Tangential velocity x component
        p.vy =  v * math.cos(angle)                   # Tangential velocity y component
        particles.append(p)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

running = True
while running:
    dt = clock.tick(FPS) / 1000.0   # Delta time in seconds
    dt = min(dt, 0.016)             # Cap dt to avoid instability on slow frames
    currentfps = clock.get_fps()    # Current measured FPS for HUD display

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exit the main loop

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            slider_area = pygame.Rect(20, 900, 200, 120)  # Region occupied by GUI sliders
            if not slider_area.collidepoint(mx, my):       # Ignore clicks on the slider panel
                if event.button == 1:
                    # Left click: spawn a small particle with random velocity
                    p = Particle(mx, my, mass=20)
                    p.vx = random.uniform(-30, 30)
                    p.vy = random.uniform(-30, 30)
                    particles.append(p)
                elif event.button == 3:
                    # Right click: spawn a black hole at cursor position
                    bh = Particle(mx, my, mass=black_hole_mass)
                    bh.is_black_hole = True
                    bh.color = (180, 0, 255)
                    particles.append(bh)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                particles.clear()                  # C: clear all particles
            elif event.key == pygame.K_SPACE:
                # Space: spawn a burst of 30 random particles across the screen
                for _ in range(30):
                    p = Particle(
                        random.randint(100, width - 100),
                        random.randint(100, height - 100),
                        mass=random.uniform(5, 40)
                    )
                    p.vx = random.uniform(-50, 50)
                    p.vy = random.uniform(-50, 50)
                    particles.append(p)
            elif event.key == pygame.K_g:
                # G: generate a galaxy-like orbit system at screen center
                create_orbit_system(width // 2, height // 2)

        manager.process_events(event)  # Forward event to pygame_gui

    # --- Update phase ---
    for p in particles:
        p.update(dt * time_scale)  # Integrate motion for each particle

    # Compute adaptive color scale based on the fastest particle this frame
    max_speed_frame = max((p.speed for p in particles if not p.is_black_hole), default=0.0)
    # Keep the color scale from collapsing too quickly by decaying it gradually
    running_color_max = max(COLOR_MAX_SPEED, running_color_max * COLOR_DECAY, max_speed_frame)
    color_scale = max(running_color_max, 1.0)  # Avoid division by zero

    # Assign color to each non-black-hole particle based on its normalized speed
    for p in particles:
        if not p.is_black_hole:
            norm_speed = min(p.speed / color_scale, 1.0)  # Normalize to [0, 1]
            p.color = speed_to_color(norm_speed)

    apply_gravity_bh(particles, dt * time_scale)  # ← Barnes-Hut gravity solver
    handle_collisions(particles)                   # Detect and resolve collisions
    manager.update(dt)                             # Update pygame_gui state

    # Sync simulation parameters from GUI sliders
    G             = g_slider.get_current_value()
    black_hole_mass = bh_slider.get_current_value()
    time_scale    = time_scale_slider.get_current_value()

    # --- Draw phase ---
    screen.fill((0, 0, 0))  # Clear screen with black background

    for p in particles:
        if p.is_black_hole:
            draw_black_hole(screen, p)  # Special rendering for black holes
        else:
            p.draw(screen)              # Normal trail + circle rendering

    # Low FPS warning overlay
    if currentfps < 30:
        screen.blit(font.render(f'WARNING: Low FPS: {int(currentfps)}', True, (255, 0, 0)), (10, 50))

    # Particle count and control hints
    screen.blit(font.render(f'Particles: {len(particles)}', True, (180, 180, 180)), (10, 10))
    screen.blit(font.render('G: galaxy | Left click: particle | Right click: black hole | SPACE: burst | C: clear', True, (100, 100, 100)), (10, 34))

    # Slider labels displayed to the right of each slider
    small_font = pygame.font.SysFont(None, 16)
    screen.blit(small_font.render(f'G (Gravity): {G:.2f}',           True, (200, 200, 200)), (230, 903))
    screen.blit(small_font.render(f'BH Mass: {black_hole_mass:.0f}', True, (200, 200, 200)), (230, 943))
    screen.blit(small_font.render(f'Time Scale: {time_scale:.2f}x',  True, (200, 200, 200)), (230, 983))

    # Speed color scale legend in the top-right corner
    legend_x = width - 220
    legend_y = 40
    screen.blit(small_font.render('Speed color legend:', True, (220, 220, 220)), (legend_x, legend_y))
    legend_y += 24
    screen.blit(small_font.render('Dark blue: slow',    True, (0, 0, 128)),     (legend_x, legend_y))
    legend_y += 20
    screen.blit(small_font.render('Light blue: medium', True, (0, 128, 255)),   (legend_x, legend_y))
    legend_y += 20
    screen.blit(small_font.render('White: fast',        True, (255, 255, 255)), (legend_x, legend_y))
    legend_y += 20
    screen.blit(small_font.render('Yellow: very fast',  True, (255, 255, 0)),   (legend_x, legend_y))
    legend_y += 20
    screen.blit(small_font.render('Red: fastest',       True, (255, 0, 0)),     (legend_x, legend_y))

    manager.draw_ui(screen)     # Render all pygame_gui elements
    pygame.display.flip()       # Present the finished frame

pygame.quit()  # Clean up pygame on exit
