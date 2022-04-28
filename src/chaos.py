import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def lorenz(x, y, z, sigma, beta, rho):
    xp = sigma*(y-x)
    yp = x * (rho - z) - y
    zp = x*y - beta*z
    return xp, yp, zp


def get_final_pop(rate, iterations=500):
    iterations = int(iterations)
    p = np.random.random()  # Initial population
    for _ in range(iterations):
        p = (rate * p) * (1 - p)
    return p


def plot_bifurcation():
    st.header("Bifurcation diagram")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Bifurcation diagram equation")
        st.latex(r'''x_{n + 1} = rx_n(1 - x_n).''')

        min_val, max_val = st.slider("Range of rate (r)", min_value=-2.0, max_value=4.0, value=(3.0, 4.0), step=0.1)
        resolution = st.number_input("Resolution (more resolution is slower)", value=1000, step=100)

    # Population
    x = []
    y = []
    domain = np.linspace(min_val, max_val, resolution)
    for u in domain:
        x.append(u)
        y.append(get_final_pop(u))

    with col2:
        fig, axs = plt.subplots()
        axs.plot(x, y, ls='', marker='.', ms=0.5)
        st.pyplot(fig)


def plot_lorenz():
    st.header("Lorenz system")

    # Initial conditions
    x = [np.random.random()]  # Speed
    y = [np.random.random()]  # Temperature
    z = [np.random.random()]  # Energy

    # Parameters and equations
    st.write("Select the parameters of the Lorenz system")

    col1, col2, col3 = st.columns(3)

    with col1:
        lorentz_sigma = st.number_input("Sigma parameter", value=10)
        st.latex(r'''
                 \begin{align}
                 \frac{\mathrm{d}x}{\mathrm{d}t} &= \sigma (y - x)
                 \end{align}
                 ''')

    with col2:
        lorentz_rho = st.number_input("Rho parameter", value=28)
        st.latex(r'''
                 \begin{align}
                 \frac{\mathrm{d}y}{\mathrm{d}t} &= x (\rho - z) - y
                 \end{align}
                 ''')

    with col3:
        lorentz_beta = st.number_input("Beta parameter", value=2.667)
        st.latex(r'''
                 \begin{align}
                 \frac{\mathrm{d}z}{\mathrm{d}t} &= x y - \beta z
                 \end{align}
                 ''')

    dt = 0.01
    for i in range(1000):
        u, v, w = lorenz(x[-1], y[-1], z[-1], lorentz_sigma, lorentz_beta, lorentz_rho)
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
    st.title("Chaos Simulation App")
    plot_lorenz()
    plot_bifurcation()


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
