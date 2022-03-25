import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def lorenz(x, y, z, sigma, beta, rho):
    xp = sigma*(y-x)
    yp = x * (rho - z) - y
    zp = x*y - beta*z
    return xp, yp, zp


def get_final_pop(rate, iterations=1000):
    v = np.random.random()  # Población inicial

    for _ in range(iterations):
        v = (rate * v) * (1 - v)

    return v


def plot_bifurcation():
    domain = np.linspace(3, 4, 10000)
    x = []
    y = []

    for i in range(5):
        for u in domain:
            x.append(u)
            y.append(get_final_pop(u, 100))

    plt.plot(x, y, ls='', marker=',')
    plt.show()

    k = [round(i * 100) for i in y]

    plt.plot(x, k, ls='', marker=',')
    plt.show()


def plot_chaotic_grid():
    r = 4.0

    x = []
    y = []
    for i in range(1000):
        x.append(get_final_pop(r))
        y.append(get_final_pop(r))

    x = [round(i * 100) for i in x]
    y = [round(i * 100) for i in y]

    plt.plot(x, y, ls='', marker='*')
    plt.show()


def plot_lorenz():
    x = [np.random.random()]  # Speed
    y = [np.random.random()]  # Temperatura
    z = [np.random.random()]  # Cantidad de energía

    dt = 0.01
    for i in range(1000):
        u, v, w = lorenz(x[-1], y[-1], z[-1], 10, 2.667, 28)
        x.append(x[-1]+u*dt)
        y.append(y[-1]+v*dt)
        z.append(z[-1]+w*dt)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot3D(x, y, z)
    plt.show()

    fig, axs = plt.subplots(3, 1)
    u = [round(i) for i in x]
    v = [round(i) for i in y]
    w = [round(i) for i in z]
    axs[0].plot(u, v)
    axs[1].plot(u, w)
    axs[2].plot(v, w)
    plt.show()


def app():
    st.title("App in construction")
