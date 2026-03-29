import pygame        # Pygame library for rendering and input handling.
import math           # Math functions (sqrt, etc.).
import random         # Random number generation.
import numpy as np    # NumPy for vectorized gravity calculations.
import pygame_gui     # pygame_gui for the real-time control panel.

pygame.init()  # Initialize all Pygame modules.

# --- Screen settings ---
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Gravity Simulation')

clock = pygame.time.Clock()
FPS = 60  # Target frames per second.

# --- Simulation parameters (also controlled via sliders) ---
G = 0.5                      # Gravitational constant.
softening = 10               # Softening factor — prevents force from becoming infinite at close range.
black_hole_mass = 500        # Mass threshold at which a merged particle becomes a black hole.
time_scale = 1.0             # Simulation speed multiplier.
SCHWARZSCHILD_MULTIPLIER = 3 # Event horizon radius = particle radius * this multiplier.

particles = []  # List of all active particles.
font = pygame.font.SysFont(None, 20)

# --- GUI setup ---
manager = pygame_gui.UIManager((width, height))

# Sliders for real-time parameter control.
g_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 900, 200, 30),
    start_value=G, value_range=(0.1, 5.0), manager=manager)

soft_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 940, 200, 30),
    start_value=softening, value_range=(1, 50), manager=manager)

bh_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 980, 200, 30),
    start_value=black_hole_mass, value_range=(100, 2000), manager=manager)

time_scale_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(20, 1020, 200, 30),
    start_value=time_scale, value_range=(0.1, 3.0), manager=manager)


class Particle:
    def __init__(self, x, y, mass):
        self.x = x
        self.y = y
        self.mass = mass
        self.radius = max(3, int(math.sqrt(mass)))  # Visual radius scales with mass.
        self.vx = 0.0   # Velocity components.
        self.vy = 0.0
        self.ax = 0.0   # Acceleration components (set by apply_gravity, used by Verlet).
        self.ay = 0.0
        self.trail = []
        self.max_trail = 80  # Maximum number of trail points to store.
        self.is_black_hole = False
        self.color = (255, 255, 255)

    def update(self, dt):
        # Store current position in trail before moving.
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        # Velocity Verlet integration:
        # More accurate than Euler — conserves energy and keeps orbits stable.
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x  += self.vx * dt + 0.5 * self.ax * dt * dt
        self.y  += self.vy * dt + 0.5 * self.ay * dt * dt

    def draw(self, screen):
        # Draw trail — older points are more transparent.
        if self.trail:
            for i, (tx, ty) in enumerate(self.trail):
                alpha = i / len(self.trail)
                r, g, b = self.color
                color = (int(r * alpha), int(g * alpha), int(b * alpha))
                pygame.draw.circle(screen, color, (int(tx), int(ty)), max(1, self.radius // 3))

        # Draw the particle core.
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def apply_gravity(particles, dt):
    """
    Calculates gravitational forces between all particle pairs using NumPy
    vectorization instead of nested Python loops — significantly faster at
    higher particle counts.

    Also handles event horizon absorption: any particle that enters a black
    hole's event horizon is removed from the simulation.
    """
    if len(particles) < 2:
        return

    # Extract positions and masses into NumPy arrays for vectorized computation.
    pos   = np.array([[p.x, p.y] for p in particles])   # Shape: (n, 2)
    mass  = np.array([p.mass     for p in particles])    # Shape: (n,)
    is_bh = np.array([p.is_black_hole for p in particles])

    # Compute all pairwise position differences in one pass.
    # dx[i, j] = x[j] - x[i] — direction from i toward j.
    dx = pos[:, 0] - pos[:, 0:1]   # Shape: (n, n)
    dy = pos[:, 1] - pos[:, 1:2]

    # Softened distance squared — prevents infinite force at zero distance.
    dist_sq = dx**2 + dy**2 + softening**2
    dist    = np.sqrt(dist_sq)

    # Pairs involving at least one black hole get a stronger gravitational pull.
    bh_pair  = is_bh[:, None] | is_bh[None, :]
    G_matrix = np.where(bh_pair, G * 5, G)   # Shape: (n, n)

    # Newton's law of gravitation: F = G * m1 * m2 / r²
    force = G_matrix * mass[:, None] * mass[None, :] / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist

    # Sum forces from all other particles and convert to acceleration (F = ma → a = F/m).
    # Divide by dt here so that ax/ay store pure acceleration, not velocity delta.
    ax = fx.sum(axis=1) / mass * dt
    ay = fy.sum(axis=1) / mass * dt

    for i, p in enumerate(particles):
        p.ax = ax[i] / dt
        p.ay = ay[i] / dt

    # Event horizon absorption — remove particles that enter a black hole's event horizon.
    to_remove = set()
    for bh in [p for p in particles if p.is_black_hole]:
        for p in [p for p in particles if not p.is_black_hole]:
            dx_ = bh.x - p.x
            dy_ = bh.y - p.y
            if math.sqrt(dx_**2 + dy_**2) < bh.radius * SCHWARZSCHILD_MULTIPLIER:
                to_remove.add(id(p))

    particles[:] = [p for p in particles if id(p) not in to_remove]


def handle_collisions(particles):
    """
    Detects overlapping particles and merges them.
    The resulting particle inherits the combined mass and conserved momentum.
    If the merged mass exceeds the black hole threshold, it becomes a black hole.
    """
    i = 0
    while i < len(particles):
        j = i + 1
        while j < len(particles):
            p1 = particles[i]
            p2 = particles[j]

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < p1.radius + p2.radius:  # Circles are overlapping — merge them.
                big   = p1 if p1.mass >= p2.mass else p2
                small = p2 if p1.mass >= p2.mass else p1

                # Conserve linear momentum: p = m*v
                big.vx = (big.mass * big.vx + small.mass * small.vx) / (big.mass + small.mass)
                big.vy = (big.mass * big.vy + small.mass * small.vy) / (big.mass + small.mass)

                big.mass += small.mass
                big.radius = max(3, int(math.sqrt(big.mass) * 2))

                # Check if the merged particle is now massive enough to become a black hole.
                if big.mass >= black_hole_mass and not big.is_black_hole:
                    big.is_black_hole = True
                    big.color = (180, 0, 255)

                particles.remove(small)
                if small is p1:
                    i -= 1
                    break
                else:
                    continue

            j += 1
        i += 1


def draw_black_hole(screen, p):
    """
    Draws a black hole with a solid black event horizon and glowing purple
    accretion rings that fade outward.
    """
    cx, cy = int(p.x), int(p.y)
    event_horizon = p.radius * SCHWARZSCHILD_MULTIPLIER

    # Draw accretion rings from outermost to innermost.
    for ring in range(5, 0, -1):
        ring_radius = int(event_horizon + ring * 6)
        alpha_surf = pygame.Surface((ring_radius * 2, ring_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(alpha_surf, (180, 0, 255, ring * 12), (ring_radius, ring_radius), ring_radius)
        screen.blit(alpha_surf, (cx - ring_radius, cy - ring_radius))

    # Event horizon — solid black core.
    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), int(event_horizon))
    # Inner bright contour.
    pygame.draw.circle(screen, (180, 0, 255), (cx, cy), int(event_horizon), 2)
    # Outer faint contour.
    pygame.draw.circle(screen, (220, 100, 255), (cx, cy), int(event_horizon) + 2, 1)


def create_orbit_system(cx, cy):
    """
    Spawns a central black hole with 200 particles in stable circular orbits.
    Each particle's velocity is calculated using the vis-viva equation: v = sqrt(G*M/r).
    The velocity direction is perpendicular to the radius vector.
    """
    sun = Particle(cx, cy, 2000)
    sun.is_black_hole = True
    sun.color = (180, 0, 255)
    particles.append(sun)

    for _ in range(200):
        r     = random.randint(120, 450)
        angle = random.uniform(0, 2 * math.pi)

        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)

        p = Particle(x, y, random.uniform(5, 20))

        # Circular orbit velocity: v = sqrt(G * M / r)
        v = math.sqrt(G * sun.mass / r)

        # Velocity direction is perpendicular to the radius vector.
        p.vx = -v * math.sin(angle)
        p.vy =  v * math.cos(angle)

        particles.append(p)


# --- Main loop ---
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    dt = min(dt, 0.016)  # Cap dt to avoid large jumps during frame rate drops.
    currentfps = clock.get_fps()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # Ignore clicks on the slider panel area.
            slider_area = pygame.Rect(20, 900, 200, 160)
            if not slider_area.collidepoint(mx, my):
                if event.button == 1:
                    # Left click: spawn a particle with a random velocity.
                    p = Particle(mx, my, mass=20)
                    p.vx = random.uniform(-30, 30)
                    p.vy = random.uniform(-30, 30)
                    particles.append(p)
                elif event.button == 3:
                    # Right click: spawn a black hole.
                    bh = Particle(mx, my, mass=black_hole_mass)
                    bh.is_black_hole = True
                    bh.color = (180, 0, 255)
                    particles.append(bh)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                particles.clear()
            elif event.key == pygame.K_SPACE:
                # Burst spawn 30 particles at random positions and velocities.
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
                create_orbit_system(width // 2, height // 2)

        manager.process_events(event)

    # --- Update ---
    # Verlet integration requires: update positions first, then recompute accelerations.
    for p in particles:
        p.update(dt * time_scale)

    apply_gravity(particles, dt * time_scale)
    handle_collisions(particles)
    manager.update(dt)

    # Sync slider values to simulation parameters.
    G             = g_slider.get_current_value()
    softening     = soft_slider.get_current_value()
    black_hole_mass = bh_slider.get_current_value()
    time_scale    = time_scale_slider.get_current_value()

    # --- Draw ---
    screen.fill((0, 0, 0))

    for p in particles:
        if p.is_black_hole:
            draw_black_hole(screen, p)
        else:
            p.draw(screen)

    # Low FPS warning.
    if currentfps < 30:
        screen.blit(font.render(f'WARNING: Low FPS: {int(currentfps)}', True, (255, 0, 0)), (10, 50))

    # HUD.
    screen.blit(font.render(f'Particles: {len(particles)}', True, (180, 180, 180)), (10, 10))
    screen.blit(font.render('G: galaxy | Left click: particle | Right click: black hole | SPACE: burst | C: clear', True, (100, 100, 100)), (10, 34))

    # Slider labels.
    small_font = pygame.font.SysFont(None, 16)
    screen.blit(small_font.render(f'G (Gravity): {G:.2f}',          True, (200, 200, 200)), (230, 903))
    screen.blit(small_font.render(f'Softening: {softening:.1f}',    True, (200, 200, 200)), (230, 943))
    screen.blit(small_font.render(f'BH Mass: {black_hole_mass:.0f}', True, (200, 200, 200)), (230, 983))
    screen.blit(small_font.render(f'Time Scale: {time_scale:.2f}x', True, (200, 200, 200)), (230, 1023))

    manager.draw_ui(screen)
    pygame.display.flip()

pygame.quit()
