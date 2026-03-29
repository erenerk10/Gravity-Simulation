import pygame  # Pygame modülünü yükler.
import math  # Matematik fonksiyonları (sqrt vb.) için.
import random  # Rastgele sayı üretimi için.
import numpy as np  # Numpy kütüphanesi, büyük veri işlemleri için.
import pygame_gui  # Pygame GUI modülü, kullanıcı arayüzü için

pygame.init()  # Pygame modüllerini başlatır.

width, height = 1920, 1080  # Ekran boyutları (genişlik, yükseklik).
screen = pygame.display.set_mode((width, height))  # Pencereyi oluşturur.
pygame.display.set_caption('Gravity Simulation')  # Pencere başlığını ayarlar.

clock = pygame.time.Clock()  # FPS kontrolü için saat objesi.
FPS = 60  # Hedef FPS değeri.

G = 0.5  # Çekim sabiti.
softening = 10  # Softening değeri, kuvvetin sonsuza gitmesini engeller.
black_hole_mass = 500  # Kara delik oluşma kütle eşik değeri.
time_scale = 1.0  # Zaman hızı çarpanı (simülasyon hızı).
SCHWARZSCHILD_MULTIPLIER = 3  # Olay ufku yarıçap çarpanı.

particles = []  # Parçacıkların saklandığı liste.
font = pygame.font.SysFont(None, 20)  # Metin çizimi için font.

manager = pygame_gui.UIManager((width, height))  # GUI yöneticisi oluşturur.

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

class Particle:  # Parçacık sınıf tanımı.
    def __init__(self, x, y, mass):  # Başlatıcı metot.
        self.x = x  # X koordinatı.
        self.y = y  # Y koordinatı.
        self.ax = 0.0  # X ivme bileşeni.
        self.ay = 0.0  # Y ivme bileşeni.
        self.mass = mass  # Kütle.
        self.radius = max(3, int(math.sqrt(mass)))  # Görsellik için yarıçap.
        self.vx = 0.0  # X hız bileşeni.
        self.vy = 0.0  # Y hız bileşeni.
        self.trail = []  # İz listesi.
        self.max_trail = 80  # İz maksimum uzunluğu.
        self.is_black_hole = False  # Kara delik mi?
        self.color = (255, 255, 255)  # Renk.

    def update(self, dt):
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x  += self.vx * dt + 0.5 * self.ax * dt * dt
        self.y  += self.vy * dt + 0.5 * self.ay * dt * dt

    def draw(self, screen):  # Parçacığı ve izini çizer.
        if self.trail:
            for i, (tx, ty) in enumerate(self.trail):  # İz parçacıklarını döngü.
                alpha = i / len(self.trail)  # Eski iz daha saydam olur.
                r, g, b = self.color
                color = (int(r * alpha), int(g * alpha), int(b * alpha))
                pygame.draw.circle(screen, color, (int(tx), int(ty)), max(1, self.radius // 3))

        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)  # Parçacık çekirdek.


class QuadTree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary  # (x, y, w, h)
        self.capacity = capacity
        self.points = []
        self.mass = 0
        self.center_of_mass = (0, 0)
        self.divided = False

    def subdivide(self):
        x, y, w, h = self.boundary
        self.northwest = QuadTree((x, y, w/2, h/2), self.capacity)
        self.northeast = QuadTree((x + w/2, y, w/2, h/2), self.capacity)
        self.southwest = QuadTree((x, y + h/2, w/2, h/2), self.capacity)
        self.southeast = QuadTree((x + w/2, y + h/2, w/2, h/2), self.capacity)
        self.divided = True
        # Redistribute existing points
        for p in self.points:
            if self.northwest.insert(p): continue
            if self.northeast.insert(p): continue
            if self.southwest.insert(p): continue
            if self.southeast.insert(p): continue
        self.points = []
        # Update mass and center of mass
        self.update_from_children()

    def insert(self, point):
        if not self.contains(point):
            return False
        if len(self.points) < self.capacity:
            self.points.append(point)
            self.update_mass(point)
            return True
        if not self.divided:
            self.subdivide()
        if self.northwest.insert(point): return True
        if self.northeast.insert(point): return True
        if self.southwest.insert(point): return True
        if self.southeast.insert(point): return True
        return False

    def contains(self, point):
        x, y = point.x, point.y
        bx, by, bw, bh = self.boundary
        return bx <= x < bx + bw and by <= y < by + bh

    def update_mass(self, point):
        total_mass = self.mass + point.mass
        cmx = (self.center_of_mass[0] * self.mass + point.x * point.mass) / total_mass
        cmy = (self.center_of_mass[1] * self.mass + point.y * point.mass) / total_mass
        self.center_of_mass = (cmx, cmy)
        self.mass = total_mass

    def calculate_force(self, point, theta, G, softening):
        if len(self.points) == 0:
            return 0, 0
        dx = self.center_of_mass[0] - point.x
        dy = self.center_of_mass[1] - point.y
        dist_sq = dx**2 + dy**2 + softening**2
        dist = math.sqrt(dist_sq)
        if not self.divided or self.boundary[2] / dist < theta:
            # Approximation
            force = G * self.mass * point.mass / dist_sq
            fx = force * dx / dist
            fy = force * dy / dist
            return fx, fy
        else:
            # Recurse
            fx, fy = 0, 0
            fx += self.northwest.calculate_force(point, theta, G, softening)[0]
            fy += self.northwest.calculate_force(point, theta, G, softening)[1]
            fx += self.northeast.calculate_force(point, theta, G, softening)[0]
            fy += self.northeast.calculate_force(point, theta, G, softening)[1]
            fx += self.southwest.calculate_force(point, theta, G, softening)[0]
            fy += self.southwest.calculate_force(point, theta, G, softening)[1]
            fx += self.southeast.calculate_force(point, theta, G, softening)[0]
            fy += self.southeast.calculate_force(point, theta, G, softening)[1]
            return fx, fy


    def update_from_children(self):
        if not self.divided:
            return
        children = [self.northwest, self.northeast, self.southwest, self.southeast]
        total_mass = sum(c.mass for c in children)
        if total_mass == 0:
            self.mass = 0
            self.center_of_mass = (0, 0)
            return
        cmx = sum(c.center_of_mass[0] * c.mass for c in children) / total_mass
        cmy = sum(c.center_of_mass[1] * c.mass for c in children) / total_mass
        self.center_of_mass = (cmx, cmy)
        self.mass = total_mass


def apply_gravity(particles, dt):
    if len(particles) < 2:
        return

    pos    = np.array([[p.x, p.y] for p in particles])
    mass   = np.array([p.mass     for p in particles])
    is_bh  = np.array([p.is_black_hole for p in particles])

    dx = pos[:, 0] - pos[:, 0:1]
    dy = pos[:, 1] - pos[:, 1:2]

    dist_sq = dx**2 + dy**2 + softening**2
    dist    = np.sqrt(dist_sq)

    # Kara delik içeren çiftlerde G * 10, diğerlerinde G * 1
    bh_pair = is_bh[:, None] | is_bh[None, :]
    G_matrix = np.where(bh_pair, G * 5, G)

    force = G_matrix * mass[:, None] * mass[None, :] / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist

    ax = fx.sum(axis=1) / mass * dt
    ay = fy.sum(axis=1) / mass * dt

    for i, p in enumerate(particles):
        p.ax = ax[i] / dt
        p.ay = ay[i] / dt

    # Olay ufku yutma
    to_remove = set()
    for bh in [p for p in particles if p.is_black_hole]:
        for p in [p for p in particles if not p.is_black_hole]:
            dx_ = bh.x - p.x
            dy_ = bh.y - p.y
            if math.sqrt(dx_**2 + dy_**2) < bh.radius * SCHWARZSCHILD_MULTIPLIER:
                to_remove.add(id(p))

    particles[:] = [p for p in particles if id(p) not in to_remove]

def handle_collisions(particles):  # Çarpışmaları işler.
    i = 0
    while i < len(particles):
        j = i + 1
        while j < len(particles):
            p1 = particles[i]
            p2 = particles[j]

            dx = p2.x - p1.x  # X farkı.
            dy = p2.y - p1.y  # Y farkı.
            dist = math.sqrt(dx * dx + dy * dy)  # Gerçek uzaklık.

            if dist < p1.radius + p2.radius:  # Eğer daireler çakışıyorsa.
                big = p1 if p1.mass >= p2.mass else p2  # Büyükünü seç.
                small = p2 if p1.mass >= p2.mass else p1  # Küçüğünü seç.

                big.vx = (big.mass * big.vx + small.mass * small.vx) / (big.mass + small.mass)  # Momentum birleşim x.
                big.vy = (big.mass * big.vy + small.mass * small.vy) / (big.mass + small.mass)  # Momentum birleşim y.

                big.mass += small.mass  # Kütleleri birleştir.
                big.radius = max(3, int(math.sqrt(big.mass) * 2))  # Yarıçap güncelle.

                if big.mass >= black_hole_mass and not big.is_black_hole:  # Kara delik oluşumu.
                    big.is_black_hole = True
                    big.color = (180, 0, 255)

                particles.remove(small)  # Küçüğü sil.
                if small is p1:
                    i -= 1  # İndeks düzelt.
                    break
                else:
                    continue

            j += 1
        i += 1


def draw_black_hole(screen, p):  # Kara deliği çizer.
    cx, cy = int(p.x), int(p.y)
    event_horizon = p.radius * SCHWARZSCHILD_MULTIPLIER  # Olay ufku yarıçap.

    for ring in range(5, 0, -1):
        ring_radius = int(event_horizon + ring * 6)  # Halka yarıçapı.
        alpha_surf = pygame.Surface((ring_radius * 2, ring_radius * 2), pygame.SRCALPHA)  # Saydam yüzey.
        pygame.draw.circle(alpha_surf, (180, 0, 255, ring * 12), (ring_radius, ring_radius), ring_radius)  # Solar halka.
        screen.blit(alpha_surf, (cx - ring_radius, cy - ring_radius))

    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), int(event_horizon))  # Olay ufku siyah.
    pygame.draw.circle(screen, (180, 0, 255), (cx, cy), int(event_horizon), 2)  # İç parlak kontur.
    pygame.draw.circle(screen, (220, 100, 255), (cx, cy), int(event_horizon) + 2, 1)  # Dış çizgiler.



def create_orbit_system(cx, cy):
    # Ortaya büyük kütle
    sun = Particle(cx, cy, 2000)
    sun.is_black_hole = True
    sun.color = (180, 0, 255)
    particles.append(sun)

    # Etrafına yörüngede parçacıklar
    for _ in range(200):
        r = random.randint(120, 450)
        angle = random.uniform(0, 2 * math.pi)

        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)

        p = Particle(x, y, random.uniform(5, 20))

        # Yörünge hızı
        v = math.sqrt(G * sun.mass / r)

        # Hız yönü (yarıçapa dik)
        p.vx = -v * math.sin(angle)
        p.vy =  v * math.cos(angle)

        particles.append(p)


running = True  # Ana döngü.
while running:
    dt = clock.tick(FPS) / 1000.0  # Delta-time hesaplama.
    dt = min(dt, 0.16)  # Ani FPS düşüşlerinde büyük dt'yi sınırlama.
    currentfps = clock.get_fps()  # Anlık FPS ölçümü.

    for event in pygame.event.get():  # Eski olayları al.
        if event.type == pygame.QUIT:
            running = False  # Kapat.
        elif event.type == pygame.MOUSEBUTTONDOWN:  # Fare tıklama.
            mx, my = pygame.mouse.get_pos()
            # Slider'lar üzerinde tıklandı mı kontrol et
            slider_area = pygame.Rect(20, 900, 200, 160)  # Tüm slider'lar için alan
            if not slider_area.collidepoint(mx, my):
                if event.button == 1:
                    p = Particle(mx, my, mass=20)
                    p.vx = random.uniform(-30, 30)
                    p.vy = random.uniform(-30, 30)
                    particles.append(p)
                elif event.button == 3:
                    bh = Particle(mx, my, mass=black_hole_mass)
                    bh.is_black_hole = True
                    bh.color = (180, 0, 255)
                    particles.append(bh)
        elif event.type == pygame.KEYDOWN:  # Klavye tuşları.
            if event.key == pygame.K_c:
                particles.clear()  # Temizle.
            elif event.key == pygame.K_SPACE:
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
        
        manager.process_events(event)   # Her olay için manager'a ilet.

    for p in particles:
       p.update(dt * time_scale)  # Parçacığı hareket ettir (zaman hızıyla çarpılır).
 
    apply_gravity(particles, dt * time_scale)  # Çekim güncelle (zaman hızıyla çarpılır).
    handle_collisions(particles)  # Çarpışma işle.
    manager.update(dt)              # GUI güncelle

    G = g_slider.get_current_value()
    softening = soft_slider.get_current_value()
    black_hole_mass = bh_slider.get_current_value()
    time_scale = time_scale_slider.get_current_value()

    screen.fill((0, 0, 0))  # Ekranı temizle.
    for p in particles:
        if p.is_black_hole:
            draw_black_hole(screen, p)  # Kara deliği çiz.
        else:
            p.draw(screen)  # Diğer parçacıkları çiz.
    
    if currentfps < 30:
        screen.blit(font.render(f'WARNING: Low FPS: {int(currentfps)}', True, (255, 0, 0)), (10, 50))  # FPS uyarısı.

    screen.blit(font.render(f'Particles: {len(particles)}', True, (180, 180, 180)), (10, 10))  # Durum bilgisi.
    screen.blit(font.render('G: galaxy | Left click: particle | Right click: black hole | SPACE: burst | C: clear', True, (100, 100, 100)), (10, 34))

    # Slider etiketleri
    small_font = pygame.font.SysFont(None, 16)
    screen.blit(small_font.render(f'G (Gravity): {G:.2f}', True, (200, 200, 200)), (230, 903))
    screen.blit(small_font.render(f'Softening: {softening:.1f}', True, (200, 200, 200)), (230, 943))
    screen.blit(small_font.render(f'BH Mass: {black_hole_mass:.0f}', True, (200, 200, 200)), (230, 983))
    screen.blit(small_font.render(f'Time Scale: {time_scale:.2f}x', True, (200, 200, 200)), (230, 1023))

    manager.draw_ui(screen)         # GUI'yi çiz (EN SON, flip'ten önce)
    pygame.display.flip()  # Ekranı güncelle.

pygame.quit()  # Pygame kapat.
