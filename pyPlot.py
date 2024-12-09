import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import pandas as pd

# Orbital mechanics function
def eph(el, t, rho=False, rv=False):
    P, TE, e, a, W, w, i, K1, K2, V0 = el
    pi2 = 2 * np.pi
    gr = 180 / np.pi

    # Mean anomaly
    M = (2 * np.pi / P) * (t - TE)
    M = np.mod(M, 2 * np.pi)  # Ensure M is within [0, 2π]

    # Solve Kepler's equation for Eccentric anomaly
    E = M
    for _ in range(10):  # Iterate for convergence
        E = M + e * np.sin(E)

    # True anomaly (theta) and distance (rho)
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    rho = a * (1 - e * np.cos(E))

    result = []
    if rho:
        x = rho * (np.cos(theta) * np.cos(W) - np.sin(theta) * np.sin(W) * np.cos(i))
        y = rho * (np.cos(theta) * np.sin(W) + np.sin(theta) * np.cos(W) * np.cos(i))
        result = np.column_stack((x, y))

    return result

# Function to parse the input file
def parse_input(file):
    content = file.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    # Extract orbital elements
    obj = {}
    for line in lines:
        if '=' in line:
            key, value = map(str.strip, line.split('=', 1))
            try:
                obj[key] = float(value)
            except ValueError:
                obj[key] = value

    # Extract data table
    data_start = False
    data = []
    for line in lines:
        if data_start:
            parts = line.split()
            if len(parts) >= 4:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        elif line.startswith("Year"):
            data_start = True

    df = pd.DataFrame(data, columns=["Year", "Value", "Error"])
    return obj, df

# Plot orbital trajectory and radial velocity
def plot_orbit(el, data):
    t = np.linspace(min(data["Year"]), max(data["Year"]), 1000)
    results = eph(el, t, rho=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(results[:, 0], results[:, 1], label="Orbit")
    ax.scatter(data["Year"], data["Value"], color="red", label="Observations")
    ax.set_title("Orbital Plot")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Orbital Analysis Tool")

    file_input = st.file_uploader("Upload input file (.txt)", type=["txt"])
    if file_input:
        obj, data = parse_input(file_input)

        st.write(f"Object: {obj.get('Object', 'N/A')}")
        st.write(f"RA (deg): {obj.get('RA (deg)', 'N/A')}")
        st.write(f"Dec (deg): {obj.get('Dec (deg)', 'N/A')}")
        el = [obj.get(key, 0) for key in ["P", "TE", "e", "a", "W", "w", "i", "K1", "K2", "V0"]]

        if st.button("Plot Orbit"):
            plot_orbit(el, data)

if __name__ == "__main__":
    main()
