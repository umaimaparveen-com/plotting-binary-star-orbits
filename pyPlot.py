import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io

# Orbital mechanics function
def eph(el, t, rho=False, rv=False):
    """
    Calculates ephemeris (theta, rho) or RV for a given set of orbital elements and time.
    """
    res = np.zeros((len(t), 2))
    pi2 = 2 * np.pi
    gr = 180 / np.pi
    P, TE, e, a, W, w, i, K1, K2, V0 = el

    # Mean anomaly
    M = (2 * np.pi / P) * (t - TE)
    M = np.mod(M, 2 * np.pi)

    # Solve Kepler's equation for Eccentric anomaly
    E = M
    for _ in range(5):  # Iterative solution for better accuracy
        E = M + e * np.sin(E)

    # True anomaly (theta) and distance (rho)
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    rho = a * (1 - e * np.cos(E))

    # If RV is requested, compute radial velocity
    if rv:
        RV = K1 * (np.cos(theta + w) + e * np.cos(w)) + V0
        return RV

    res[:, 0] = theta * gr  # Convert to degrees
    res[:, 1] = rho
    return res

# Read data from file
def readtxt(file):
    """
    Reads orbital element data and observational data from a .txt file.
    Args:
        file: Uploaded file object.
    Returns:
        obj: Dictionary with orbital elements and observational data.
    """
    obj = {}
    file_content = io.StringIO(file.getvalue().decode("utf-8"))
    lines = file_content.readlines()

    observational_data = []
    orbital_elements = [
        'P', 'TE', 'e', 'a', 'W', 'w', 'i', 'K1', 'K2', 'V0'
    ]  # Expected orbital elements

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "Object:" in line:
            obj['name'] = line.split(":", 1)[1].strip()

        elif "=" in line:
            key, value = line.split("=", 1)
            try:
                obj[key.strip()] = float(value.strip())
            except ValueError:
                obj[key.strip()] = value.strip()

        else:
            # Parse observational data
            parts = line.split()
            if len(parts) >= 4:
                observational_data.append([float(parts[0]), float(parts[1]), float(parts[2])])

    obj['el'] = [obj.get(key, 0) for key in orbital_elements]
    obj['data'] = np.array(observational_data)
    return obj

# Plot orbital trajectory and radial velocity
def orbplot(obj):
    el = obj['el']
    data = obj['data']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Orbital plot
    t = np.linspace(0, el[0], 1000)  # One orbital period
    result = eph(el, t, rho=True)
    theta, rho = result[:, 0], result[:, 1]
    ax[0].plot(rho * np.cos(np.radians(theta)), rho * np.sin(np.radians(theta)), label="Orbit")
    ax[0].set_title('Orbital Plot')
    ax[0].set_xlabel('x (AU)')
    ax[0].set_ylabel('y (AU)')
    ax[0].grid(True)
    ax[0].legend()

    # Radial velocity plot
    rv = eph(el, t, rv=True)
    ax[1].plot(t, rv, label="Radial Velocity")
    ax[1].set_title('Radial Velocity Plot')
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('RV (km/s)')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Display observational data
    st.write("Observational Data (Year, Value, Error):")
    st.write(data)

# Main application
def main():
    st.title("Orbital Analysis Tool")

    # Input file
    file_input = st.file_uploader("Upload input file (.txt)", type=["txt"])

    if file_input is not None:
        # Read the input file
        obj = readtxt(file_input)

        el_fit, el_err_fit = fitorb(el, yy, err, fita)

        # Calculate ephemeris and radial velocity
        t = np.linspace(min(time), max(time), 1000)
        result = eph(el_fit, t, rho=True, rv=True)
        theta, rho, rv = result[:, 0], result[:, 1], result[:, 2]
    
        # Plot orbital trajectory and radial velocity curve
        orbplot(obj, el_fit, el_err_fit, fixel)

        # Display parsed orbital elements
        st.write(f"Object: {obj.get('name', 'N/A')}")
        st.write(f"RA (deg): {obj.get('RA (deg)', 'N/A')}")
        st.write(f"Dec (deg): {obj.get('Dec (deg)', 'N/A')}")
        st.write("Orbital Elements:", obj.get('el', 'N/A'))

        # Plot the data
        if st.button("Plot Orbital Parameters"):
            orbplot(obj)

if __name__ == "__main__":
    main()
