import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# READ ME ===============================================================
# Lines 13 - 86 and 139 - 145 are used for simulation anc calculation
# Rest of the lines are for plotting and animation
# =======================================================================

def euler_step_function(alpha, h, mass, g, Ib, dt, sx, vx, omega, b, radius):
    acc_b = g * math.sin(alpha) / (1 + Ib / (mass * radius ** 2))
    eps = acc_b / radius

    dsx = vx * dt
    dvx = acc_b * dt
    mx = sx * math.cos(-alpha) - radius * math.sin(-alpha)
    my = sx * math.sin(-alpha) + radius * math.cos(-alpha) + h

    db = omega * dt
    dw = eps * dt
    ax = radius * math.cos(b) + mx
    ay = radius * math.sin(b) + my

    sx += dsx
    vx += dvx
    b -= db
    omega += dw

    return sx, vx, omega, b, ax, ay, mx, my


def mid_point_step_function(alpha, h, mass, g, Ib, dt, sx, vx, omega, b, radius):
    acc_b = g * math.sin(alpha) / (1 + Ib / (mass * radius ** 2))
    eps = acc_b / radius
    euler_omega = omega + eps * dt / 2
    euler_vx = vx + acc_b * dt / 2

    dsx = euler_vx * dt
    dvx = acc_b * dt
    mx = sx * math.cos(-alpha) - radius * math.sin(-alpha)
    my = sx * math.sin(-alpha) + radius * math.cos(-alpha) + h

    db = euler_omega * dt
    dw = eps * dt
    ax = radius * math.cos(b) + mx
    ay = radius * math.sin(b) + my

    sx += dsx
    vx += dvx
    b -= db
    omega += dw

    return sx, vx, omega, b, ax, ay, mx, my


def simulate(alpha, h, mass, g, Ib, dt, radius):
    t = 0
    actualX, actualY, middleX, middleY = [], [], [], []
    energies = {"KE": [], "PE": [], "TE": []}
    sx = vx = w = 0
    b = math.radians(90)

    while True:  # Simulation time
        sx, vx, w, b, ax, ay, mx, my = mid_point_step_function(alpha, h, mass, g, Ib, dt, sx, vx, w, b, radius)
        if ay <= 0:
            break
        middleX.append(mx)
        middleY.append(my)
        actualX.append(ax)
        actualY.append(ay)

        KE = 0.5 * mass * vx ** 2 + 0.5 * Ib * w ** 2
        PE = mass * g * my
        TE = KE + PE
        energies["KE"].append(KE)
        energies["PE"].append(PE)
        energies["TE"].append(TE)

        t += dt

    return actualX, actualY, middleX, middleY, energies


def animate_simulation_with_mass_points_and_line(actualX, actualY, middleX, middleY, alpha, radius, slope_end_x):
    fig, ax = plt.subplots()
    ax.set_xlim((min(middleX) - 10, slope_end_x))
    ax.set_ylim((min(middleY) - 10, max(middleY) + 10 + radius))
    ax.set_aspect('equal', adjustable='box')


    slope_y = h - np.linspace(0, slope_end_x, 100) * math.tan(alpha)
    ax.plot(np.linspace(0, slope_end_x, 100), slope_y, 'g-', label='Slope',color='black')
    ax.fill_between(np.linspace(0, slope_end_x, 100), 0, slope_y, color='lightgrey', alpha=0.5, label='Under Slope')


    circle = Circle((0, 0), radius, fill=False, color='blue')
    ax.add_patch(circle)

    mass_points, = ax.plot([], [], 'ro', markersize=2, label='Center of Mass Path', color='darkslategray')
    rotation_line, = ax.plot([], [], 'r-', label='Rotation Line', color='red')
    wave_points, = ax.plot([], [], 'r-', markersize=1, label='Rotation of Fixed Path', color='orange')

    def init():
        circle.center = (middleX[0], middleY[0])
        rotation_line.set_data([], [])
        mass_points.set_data([], [])
        wave_points.set_data([], [])
        return circle, rotation_line, mass_points, wave_points

    def animate(i):
        circle.center = (middleX[i], middleY[i])
        rotation_line.set_data([middleX[i], actualX[i]], [middleY[i], actualY[i]])
        mass_points.set_data(middleX[:i + 1], middleY[:i + 1])
        wave_points.set_data(actualX[:i + 1], actualY[:i + 1])
        return circle, rotation_line, mass_points, wave_points

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(middleX), interval=50, blit=True)
    plt.legend()
    plt.title("Improved Eulers Method")
    plt.show()


def plot_energies(energies, dt):
    plt.figure(figsize=(10, 6))
    time = np.arange(0, len(energies["KE"]) * dt, dt)
    plt.plot(time, energies["KE"], label="Kinetic Energy")
    plt.plot(time, energies["PE"], label="Potential Energy")
    plt.plot(time, energies["TE"], label="Total Energy")
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Energy over Time')
    plt.legend()
    plt.show()


alpha = math.radians(20)
h = 100
mass = 1
radius = 5
g = 9.81
Ib = 2 / 5 * mass * radius ** 2
dt = 0.1


actualX, actualY, middleX, middleY, energies = simulate(alpha, h, mass, g, Ib, dt, radius)
slope_end_x = max(middleX) + 10
animate_simulation_with_mass_points_and_line(actualX, actualY, middleX, middleY, alpha, radius, slope_end_x)
plot_energies(energies, dt)
