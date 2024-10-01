import PyDSTool as dst
import numpy as np
import matplotlib.pyplot as plt

# Define the ODE
def R_inf(x, w, I):
    return 1 / (1 + np.exp(-w * x - I))

# ODE parameters
params = {
    'w': 1.0,  # This is a sample value; adjust as needed
    'I': 0.0   # Continuation will vary this parameter
}

# Initial guess for fixed points
ic_guess = {'r': 0.5}

# Define the ODE system
DSargs = dst.args(name='PopulationRate')
DSargs.varspecs = {'r': '-r + R_inf(r, w, I)'}
DSargs.pars = params
DSargs.ics = ic_guess

# Create the Generator
ODE = dst.Generator.Vode_ODEsystem(DSargs)

# Set up continuation class
PC = dst.ContClass(ODE)

# Perform continuation
PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars = ['I']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 500
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True
PC.newCurve(PCargs)

print("Computing curve...")
PC['EQ1'].forward()
PC['EQ1'].backward()

# Plot bifurcation diagram
plt.figure(figsize=(8, 4))
PC.display(['I', 'r'], stability=True, figure=3)
plt.title('Bifurcation diagram: r vs I')
plt.xlabel('Drive (I)')
plt.ylabel('Population rate (r)')

# Extract data for the I/O curve
curve_data = PC['EQ1'].sol
I_vals = curve_data['I']
r_vals = curve_data['r']

# Plot I/O curve
plt.figure(figsize=(8, 4))
plt.plot(I_vals, r_vals, label='I/O Curve')
plt.title('I/O Curve in I-w Phase Plane')
plt.xlabel('Drive (I)')
plt.ylabel('Population rate (r)')
plt.legend()
plt.show()
