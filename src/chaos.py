import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def lorenz(x, y, z, sigma, beta, rho):
    xp = sigma*(y-x)
    yp = x * (rho - z) - y
    zp = x*y - beta*z
    return xp, yp, zp


def get_final_pop(rate, iterations=1000):
    p = np.random.random()  # Initial population
    for _ in range(iterations):
        p = (rate * p) * (1 - p)
    return p


def plot_bifurcation():
    domain = np.linspace(3, 4, 10000)
    x = []
    y = []

    for i in range(5):
        for u in domain:
            x.append(u)
            y.append(get_final_pop(u, 100))
    k = [round(i * 100) for i in y]

    col1, col2 = st.columns(2)

    with col1:
        fig, axs = plt.subplots()
        axs.plot(x, y, ls='', marker=',')
        st.pyplot(fig)

    with col2:
        fig, axs = plt.subplots()
        axs.plot(x, k, ls='', marker=',')
        st.pyplot(fig)


def plot_lorenz():
    x = [np.random.random()]  # Speed
    y = [np.random.random()]  # Temperature
    z = [np.random.random()]  # Energy

    dt = 0.01
    for i in range(1000):
        u, v, w = lorenz(x[-1], y[-1], z[-1], 10, 2.667, 28)
        x.append(x[-1]+u*dt)
        y.append(y[-1]+v*dt)
        z.append(z[-1]+w*dt)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection='3d')
        ax.plot3D(x, y, z)
        st.pyplot(fig)

    with col2:
        fig, axs = plt.subplots(3, 1)
        u = [round(i) for i in x]
        v = [round(i) for i in y]
        w = [round(i) for i in z]
        axs[0].plot(u, v)
        axs[1].plot(u, w)
        axs[2].plot(v, w)
        st.pyplot(fig)


def app():
    st.title("Chaos Simulator")

    st.header("Bifurcation diagram")
    plot_bifurcation()

    st.header("Lorenz system")
    plot_lorenz()


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
