import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

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
    # Arrays for Midpoint method
    actualX_mp, actualY_mp, middleX_mp, middleY_mp = [], [], [], []
    # Arrays for Euler method
    actualX_eu, actualY_eu, middleX_eu, middleY_eu = [], [], [], []
    energies_mp = {"KE": [], "PE": [], "TE": []}
    energies_eu = {"KE": [], "PE": [], "TE": []}
    sx_mp = sx_eu = vx_mp = vx_eu = w_mp = w_eu = 0
    b_mp = b_eu = math.radians(90)

    while True:  # Simulation time
        # Midpoint method step
        sx_mp, vx_mp, w_mp, b_mp, ax_mp, ay_mp, mx_mp, my_mp = mid_point_step_function(alpha, h, mass, g, Ib, dt, sx_mp, vx_mp, w_mp, b_mp, radius)
        # Euler method step
        sx_eu, vx_eu, w_eu, b_eu, ax_eu, ay_eu, mx_eu, my_eu = euler_step_function(alpha, h, mass, g, Ib, dt, sx_eu, vx_eu, w_eu, b_eu, radius)

        if ay_mp <= 0 and t > 0:  # Stop condition
            break

        # Update Midpoint method results
        middleX_mp.append(mx_mp)
        middleY_mp.append(my_mp)
        actualX_mp.append(ax_mp)
        actualY_mp.append(ay_mp)

        KE_mp = 0.5 * mass * vx_mp ** 2 + 0.5 * Ib * w_mp ** 2
        PE_mp = mass * g * my_mp
        TE_mp = KE_mp + PE_mp
        energies_mp["KE"].append(KE_mp)
        energies_mp["PE"].append(PE_mp)
        energies_mp["TE"].append(TE_mp)

        # Update Euler method results
        middleX_eu.append(mx_eu)
        middleY_eu.append(my_eu)
        actualX_eu.append(ax_eu)
        actualY_eu.append(ay_eu)

        KE_eu = 0.5 * mass * vx_eu ** 2 + 0.5 * Ib * w_eu ** 2
        PE_eu = mass * g * my_eu
        TE_eu = KE_eu + PE_eu
        energies_eu["KE"].append(KE_eu)
        energies_eu["PE"].append(PE_eu)
        energies_eu["TE"].append(TE_eu)

        t += dt

    return actualX_mp, actualY_mp, middleX_mp, middleY_mp, energies_mp, actualX_eu, actualY_eu, middleX_eu, middleY_eu, energies_eu

def plot_energies(energies_mp, energies_eu, dt):
    plt.figure(figsize=(12, 6))

    # Find the minimum length to ensure matching dimensions
    min_length_mp = min(len(energies_mp["KE"]), len(energies_mp["PE"]), len(energies_mp["TE"]))
    min_length_eu = min(len(energies_eu["KE"]), len(energies_eu["PE"]), len(energies_eu["TE"]))
    min_length = min(min_length_mp, min_length_eu)
    time = np.arange(0, min_length * dt, dt)[:min_length]

    # Ensure to slice each energy array to min_length for plotting
    plt.plot(time, energies_mp["KE"][:min_length], label="Kinetic Energy (Midpoint)", linestyle='-', color='blue')
    plt.plot(time, energies_mp["PE"][:min_length], label="Potential Energy (Midpoint)", linestyle='-', color='green')
    plt.plot(time, energies_mp["TE"][:min_length], label="Total Energy (Midpoint)", linestyle='-', color='red')

    plt.plot(time, energies_eu["KE"][:min_length], label="Kinetic Energy (Euler)", linestyle='--', color='blue')
    plt.plot(time, energies_eu["PE"][:min_length], label="Potential Energy (Euler)", linestyle='--', color='green')
    plt.plot(time, energies_eu["TE"][:min_length], label="Total Energy (Euler)", linestyle='--', color='red')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Energy over Time for Midpoint and Euler Methods')
    plt.legend()
    plt.show()




# Simulation parameters (these parameters can remain unchanged from your original setup)
alpha = math.radians(30)
h = 200
mass = 1
radius = 5
g = 9.81
Ib = 2 / 3 * mass * radius ** 2
dt = 0.1

# Run the simulation for both methods
results = simulate(alpha, h, mass, g, Ib, dt, radius)
actualX_mp, actualY_mp, middleX_mp, middleY_mp, energies_mp, actualX_eu, actualY_eu, middleX_eu, middleY_eu, energies_eu = results

# Plot the energies for comparison
plot_energies(energies_mp, energies_eu, dt)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon


# Assuming simulation has been run and variables are filled:
# actualX_mp, actualY_mp, middleX_mp, middleY_mp, energies_mp
# actualX_eu, actualY_eu, middleX_eu, middleY_eu, energies_eu

slope_end_x = max(max(actualX_mp), max(actualX_eu)) + 10

def setup_plot():
    fig, ax = plt.subplots()
    ax.set_xlim((min(min(middleX_mp), min(middleX_eu)) - 10, slope_end_x))
    ax.set_ylim((min(min(middleY_mp), min(middleY_eu)) - 10, max(max(middleY_mp), max(middleY_eu)) + 10))
    ax.set_aspect('equal', adjustable='box')

    # Draw slope and fill under the slope
    slope_x = np.linspace(0, slope_end_x, 100)
    slope_y = h - slope_x * math.tan(alpha)
    ax.plot(slope_x, slope_y, 'g-', label='Slope', color='black')
    ax.fill_between(slope_x, 0, slope_y, color='lightgrey', alpha=0.5)

    return fig, ax


def animate(i):
    # Update the circle position for midpoint method
    circle_mp.center = (middleX_mp[i], middleY_mp[i])
    # Update the circle position for Euler method
    circle_eu.center = (middleX_eu[i], middleY_eu[i])

    # Update lines for midpoint and Euler methods
    line_mp.set_data([middleX_mp[i], actualX_mp[i]], [middleY_mp[i], actualY_mp[i]])
    line_eu.set_data([middleX_eu[i], actualX_eu[i]], [middleY_eu[i], actualY_eu[i]])

    return circle_mp, circle_eu, line_mp, line_eu


fig, ax = setup_plot()

# Initialize circles and lines for both methods
circle_mp = Circle((0, 0), radius, fill=False, color='blue', label='Midpoint Method')
circle_eu = Circle((0, 0), radius, fill=False, color='red', linestyle='--', label='Euler Method')
ax.add_patch(circle_mp)
ax.add_patch(circle_eu)

line_mp, = ax.plot([], [], 'blue', label='Midpoint Rotation Line')
line_eu, = ax.plot([], [], 'red', linestyle='--', label='Euler Rotation Line')

# Creating the animation
ani = FuncAnimation(fig, animate, frames=len(middleX_mp), interval= 80, blit=True)

plt.legend()
plt.show()
