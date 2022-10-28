import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
import queue
import math

class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, g, radius=0.01, styles=None):
        """Initialize the particle's position, velocity, and radius.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor.

        """

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.g = g
        self.radius = radius
        self.mass = self.radius**3
        self.epoch = 0

        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'edgecolor': 'b', 'fill': False}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.r[0]
    @x.setter
    def x(self, value):
        self.r[0] = value
    @property
    def y(self):
        return self.r[1]
    @y.setter
    def y(self, value):
        self.r[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value

    # Instantaneous position (x,y) at time t
    def pos(self, t):
    	return self.r + self.v * t + self.g * t * t / 2.0

    # Instantaneous velocity (vx,vy) at time t
    def vel(self, t):
    	return self.v + self.g * t

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle


class Simulation:
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: 0 <= x < 1, 0 <= y < 1.

    """

    ParticleClass = Particle

    def __init__(self, n, g=0.0, radius=0.01, styles=None):
        """Initialize the simulation with n Particles with radii radius.

        radius can be a single value or a sequence with n values.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor when drawing
        the Particles.

        """

        self.dt = 0.01
        self.t = 0.0
        self.g = np.array((0.0, -g))
        self.events = queue.PriorityQueue()
        self.events.put((self.dt, -1, 0, -1, 0))
        self.init_particles(n, radius, styles)

    def place_particle(self, rad, styles):
        # Choose x, y so that the Particle is entirely inside the
        # domain of the simulation.
        x, y = rad + (1 - 2*rad) * np.random.random(2)
        # Choose a random velocity (within some reasonable range of
        # values) for the Particle.
        vx, vy = 0.1 * np.random.randn(2)
        particle = self.ParticleClass(x, y, vx, vy, self.g, rad, styles)
        # Check that the Particle doesn't overlap one that's already
        # been placed.
        for p2 in self.particles:
            if p2.overlaps(particle):
                break
        else:
            self.particles.append(particle)
            return True
        return False

    def init_particles(self, n, radius, styles=None):
        """Initialize the n Particles of the simulation.

        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.

        """

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):
            # Try to find a random initial position for this particle.
            while not self.place_particle(rad, styles):
                pass

        self.add_all_collisions()

    # Predict the next collision event between particles i and j
    def predict_collision(self, i, j):
        p = self.particles[i]
        q = self.particles[j]
        r = q.r - p.r
        v = q.v - p.v
        b = 2.0 * np.dot(r, v)
        if b >= 0.0:
            return
        radius = p.radius + q.radius
        a = np.dot(v, v)
        if a == 0.0:
            return
        c = np.dot(r, r) - radius*radius
        delta = b*b - 4.0 * a * c
        if delta < 0.0:
            return
        t = (- b - math.sqrt(delta)) / (2.0 * a)
        if t < self.t:
       	    return
       	self.events.put((t, i, p.epoch, j, q.epoch))
        
    # Predict the next collision event between particle i and a wall (horizontal or vertical)
    def predict_wall_collision(self, i):
        p = self.particles[i]
        tx = math.inf
        if p.vx > 0.0:
            tx = (1.0 - p.radius - p.x) / p.vx
        elif p.vx < 0.0:
            tx = (0.0 + p.radius - p.x) / p.vx

        ty = math.inf
        if self.g[1] == 0.0:
            if p.vy > 0.0:
                ty = (1.0 - p.radius - p.y) / p.vy
            elif p.vy < 0.0:
                ty = (0.0 + p.radius - p.y) / p.vy
        else:
            a = self.g[1] / 2.0
            b = p.vy
            c = p.y - (1.0 - p.radius)
            delta = b * b - 4.0 * a * c
            if p.vel(self.t)[1] > 0.0 and delta > 0.0:
                ty = (- b + math.sqrt(delta)) / (2.0 * a)
            else:
                c = p.y - (0.0 + p.radius)
                delta = b * b - 4.0 * a * c
                ty = (- b - math.sqrt(delta)) / (2.0 * a)
        
        if tx < ty:
            self.events.put((tx, i, p.epoch, -1, 0))
        else:
            self.events.put((ty, i, p.epoch, -2, 0))

    # Recompute the next collision events for particle i (after the current collision)
    def update_collisions(self, i):
        self.particles[i].epoch += 1
        self.predict_wall_collision(i)
        for j, q in enumerate(self.particles):
            if j == i:
                continue
            self.predict_collision(i, j)

    # Predict the first collision events for all particles at the start of the simulation
    def add_all_collisions(self):
        for i, p in enumerate(self.particles):
            self.predict_wall_collision(i)
            for j, q in enumerate(self.particles):
                if j <= i:
                    continue
                self.predict_collision(i, j)

    def advance_animation(self):
        """Advance the animation by dt, returning the updated Circles list."""
        while True:
            # Get the next event from the queue
            self.t, i, ei, j, ej = self.events.get()
            if i == -1:
                # Drawing event: periodically draw after time dt
                self.events.put((self.t + self.dt, -1, 0, -1, 0))
                break
            elif j == -1:
                # x collision against a wall (horizontal flip)
                p = self.particles[i]
                if ei == p.epoch:
                    pos = p.pos(self.t)
                    x = pos[0]
                    p.x = 2.0 * x - p.x
                    p.vx = -p.vx
                    circle = Circle(xy=(pos[0]+math.copysign(p.radius, -p.vx),pos[1]), radius=0.005, color='gold')
                    self.ax.add_patch(circle)
                    if len(self.circles) > len(self.particles) + 5:
                        c = self.circles.pop(len(self.particles))
                        c.center = (-1,-1)
                    self.circles.append(circle)
                    self.update_collisions(i)
            elif j == -2:
                # y collision against ceiling or floor (vertical flip)
                p = self.particles[i]
                if ei == p.epoch:
                    pos = p.pos(self.t)
                    vel = p.vel(self.t)
                    vel[1] = -vel[1]
                    p.v = vel - self.g * self.t
                    p.r = pos - p.v * self.t - self.g * (self.t * self.t / 2.0)
                    circle = Circle(xy=(pos[0],pos[1]+math.copysign(p.radius, -vel[1])), radius=0.005, color='gold')
                    self.ax.add_patch(circle)
                    if len(self.circles) > len(self.particles) + 5:
                        c = self.circles.pop(len(self.particles))
                        c.center = (-1,-1)
                    self.circles.append(circle)
                    self.update_collisions(i)
            else:
                # Particle-Particle collision
                p = self.particles[i]
                q = self.particles[j]
                if ei == p.epoch and ej == q.epoch:
                    rp = p.pos(self.t)
                    rq = q.pos(self.t)
                    vp = p.vel(self.t)
                    vq = q.vel(self.t)
                    r = rq - rp
                    d = math.sqrt(np.dot(r, r))
                    r = r/d
                    u = 2.0 * (np.dot(vp, r) - np.dot(vq, r)) / (1.0/p.mass + 1.0/q.mass)
                    u = u * r
                    vp = vp - u/p.mass
                    vq = vq + u/q.mass
                    p.v = vp - self.g * self.t
                    q.v = vq - self.g * self.t
                    p.r = rp - p.v * self.t - self.g * (self.t * self.t / 2.0)
                    q.r = rq - q.v * self.t - self.g * (self.t * self.t / 2.0)
                    
                    circle = Circle(xy = rp + p.radius * r, radius=0.005, color='gold')
                    self.ax.add_patch(circle)
                    if len(self.circles) > len(self.particles) + 5:
                        c = self.circles.pop(len(self.particles))
                        c.center = (-1,-1)
                    self.circles.append(circle)
                    self.update_collisions(i)
                    self.update_collisions(j)
        for i, p in enumerate(self.particles):
            self.circles[i].center = p.pos(self.t)
        return self.circles

    def init(self):
        """Initialize the Matplotlib animation."""

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax))
        return self.circles

    def animate(self, i):
        """The function passed to Matplotlib's FuncAnimation routine."""

        self.advance_animation()
        return self.circles

    def setup_animation(self):
        self.fig, self.ax = plt.subplots()
        for s in ['top','bottom','left','right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

    def save_or_show_animation(self, anim, save, filename='collision.mp4'):
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, bitrate=1800)
            anim.save(filename, writer=writer)
        else:
            plt.show()

    def do_animation(self, save=False, interval=1, filename='collision.mp4'):
        """Set up and carry out the animation of the molecular dynamics.

        To save the animation as a MP4 movie, set save=True.
        """

        self.setup_animation()
        anim = animation.FuncAnimation(self.fig, self.animate,
                init_func=self.init, frames=800, interval=interval, blit=True)
        self.save_or_show_animation(anim, save, filename)


if __name__ == '__main__':
    nparticles = 40
    radii = np.random.random(nparticles)*0.03+0.01
    styles = {'edgecolor': 'C0', 'linewidth': 2}
    sim = Simulation(nparticles, 0.0, radii, styles)
    sim.do_animation(save=False)
