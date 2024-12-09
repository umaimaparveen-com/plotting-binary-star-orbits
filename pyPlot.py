import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import io
import pandas as pd

# Orbital mechanics function
def eph(el, t, rho=False, rv=False):
    """
    Calculates ephemeris (theta, rho) or RV for a given set of orbital elements and time.
    """
    n = len(t)
    res = np.zeros((len(t), 2))
    pi2 = 2 * np.pi
    gr = 180 / np.pi
    P, TE, e, a, W, w, i, K1, K2, V0 = el

    # Mean anomaly
    M = (2 * np.pi / P) * (t - TE)
    M = np.mod(M, 2 * np.pi)  # Ensuring it's within [0, 2pi]
    
    # Solve Kepler's equation for Eccentric anomaly
    E = M  # Initial guess for E (not iterative for simplicity)
    for _ in range(5):  # Simple Newton-Raphson iteration to solve Kepler's equation
        E = M + e * np.sin(E)

    # True anomaly (theta) and distance (rho)
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    rho = a * (1 - e * np.cos(E))

    # If RV is requested, compute radial velocity
    if rv:
        RV = np.sqrt(a * (1 - e ** 2)) * np.sin(E)  # Simplified RV computation

    if np.any(rho):
        res[:, 0] = theta * gr  # Convert to degrees
        res[:, 1] = rho
        
    elif rv:
        res[:, 0] = RV
        
    return res

# Convert a decimal coordinate string to degrees
def getcoord(s):
    l = s.find('.')
    deg = int(s[:l])
    min = int(s[l+1:l+3])
    sec = float(s[l+3:])
    res = abs(deg) + min/60. + sec/3600.
    return res * np.sign(deg)

# Corrects Julian Days or years based on a reference time
def correct(data, t0):
    time = data[:, 0]
    years_to_jd = (time < 3e3) & (t0 > 3e3)
    jd_to_years = (time > 3e3) & (t0 < 3e3)
    time[years_to_jd] = 365.242198781 * (time[years_to_jd] - 1900.0) + 15020.31352
    time[jd_to_years] = 1900.0 + (time[jd_to_years] - 15020.31352) / 365.242198781

# Converts a degree value to degrees, minutes, and seconds
def sixty(scalar):
    ss = abs(3600.0 * scalar)
    mm = abs(60.0 * scalar)
    dd = abs(scalar)
    result = np.array([int(dd), int(mm % 60), int(ss % 3600)])
    if scalar < 0:
        result[result != 0] *= -1
    return result

def readtxt(file):
    """
    Reads orbital element data from an uploaded .txt file.
    
    Args:
        file: Uploaded file object.
        
    Returns:
        obj: Dictionary containing parsed data.
    """
    obj = {}
    
    # Read the content of the uploaded file as text
    file_content = io.StringIO(file.getvalue().decode("utf-8"))
    lines = file_content.readlines()

    orbital_elements = []  # List to store orbital elements
    found_orbital_elements = False  # Flag to indicate if we've found orbital elements section

    # Iterate through lines to find the orbital elements
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip comment lines starting with "#"
        if line.startswith('#'):
            continue

        # Process key-value pairs for orbital data
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()

                # Parse as float or int if possible
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep value as string if it cannot be converted

                obj[key] = value
                
                # If we have found the orbital element keys (P, TE, e, etc.), set the flag
                if key == 'P':  # When we encounter P, the orbital elements section starts
                    found_orbital_elements = True

        # Check for specific orbital elements after "P"
        if found_orbital_elements:
            # These should be the actual orbital elements, you can adjust based on the file format
            try:
                obj['el'] = [
                    obj.get('P', 0), obj.get('TE', 0), obj.get('e', 0),
                    obj.get('a', 0), obj.get('W', 0), obj.get('w', 0),
                    obj.get('i', 0), obj.get('K1', 0), obj.get('K2', 0),
                    obj.get('V0', 0)
                ]
            except KeyError as e:
                st.write(f"Missing element in file: {e}")

            break  # Break after storing orbital elements

    # If orbital elements are missing, print warning and set defaults
    if not obj.get('el'):
        st.write("Warning: Orbital elements could not be parsed correctly.")
        obj['el'] = [0] * 10  # Set default orbital elements if missing

    return obj


def orbplot(obj, el, elerr, fixel):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the orbital trajectory
    t = np.linspace(0, 1000, 1000)  # Extended time array for variations in radial velocity
    result = eph(el, t, rho=True, rv=False)
    theta, rho = result[:, 0], result[:, 1]
    
    ax[0].plot(rho * np.cos(theta), rho * np.sin(theta), label="Orbit")
    ax[0].set_title('Orbital Plot')
    ax[0].set_xlabel('x (AU)')
    ax[0].set_ylabel('y (AU)')
    ax[0].grid(True)
    ax[0].legend()

    # Plot radial velocity
    rv = eph(el, t, rv=True)
    
    if isinstance(rv, tuple):
        rv = rv[0]  # Assuming RV is the first element in the tuple
    
    ax[1].plot(t, rv, label="RV")
    ax[1].set_title('Radial Velocity Plot')
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('RV (km/s)')
    ax[1].grid(True)
    ax[1].legend()

    # Print radial velocity values to verify if they are zero
    st.write("Radial Velocity Values:", rv[:10])  # Show first 10 radial velocity values for inspection

    plt.tight_layout()
    st.pyplot(fig)

# Fit orbital elements using least squares
def fitorb(par, yy, err, fita):
    # Residual function for fitting
    def residuals(par, yy, err, fita):
        # Get the model from the eph function
        model = eph(par, fita, rho=True)
        
        # If the model is a tuple, unpack it
        if isinstance(model, tuple):
            # Assuming the model contains theta and rho
            model = model[1]  # Assuming rho is the second element
        
        # Return the residuals (observed - model) / error
        return (yy - model) / err

    # Use the residuals function to perform the least squares fitting
    from scipy.optimize import least_squares
    result = least_squares(residuals, par, args=(yy, err, fita))
    
    return result

# Save orbital results to file
def orbsave(obj, el, elerr, fixel):
    with open('orbital_results.txt', 'w') as f:
        f.write(f"Object: {obj.get('name', 'N/A')}\n")
        f.write(f"RA: {obj.get('radeg', 'N/A')} Dec: {obj.get('dedeg', 'N/A')}\n")
        f.write("Orbital Elements: " + ' '.join(map(str, el)) + "\n")
        f.write("Errors: " + ' '.join(map(str, elerr)) + "\n")
        f.write("Fixed Elements: " + ' '.join(map(str, fixel)) + "\n")

def main():
    st.title("Orbital Analysis Tool")

    # Input file
    file_input = st.file_uploader("Upload input file (.txt)", type=["txt"])

    if file_input is not None:
        # Read input file and populate variables
        obj = readtxt(file_input)

        # Display basic information with checks
        st.write(f"Object: {obj.get('name', 'N/A')}")
        st.write(f"RA (deg): {obj.get('radeg', 'N/A')}")
        st.write(f"Dec (deg): {obj.get('dedeg', 'N/A')}")
        st.write("Orbital Elements:", obj.get('el', 'N/A'))

        # Request for confirmation
        if st.button("Plot Orbital Parameters"):
            el = obj.get('el', [])
            elerr = [0] * len(el)  # Placeholder for errors
            fixel = [True] * len(el)  # Placeholder for fixed elements
            
            orbplot(obj, el, elerr, fixel)

            # Option to save results
            if st.button("Save Results"):
                orbsave(obj, el, elerr, fixel)
                st.success("Results saved to 'orbital_results.txt'.")

if __name__ == "__main__":
    main()
