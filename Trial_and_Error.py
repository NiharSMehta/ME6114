import numpy as np
import math
from scipy.optimize import minimize

# ------------------------------------
# USER INPUTS
# ------------------------------------
hip_abduction_angle_deg = 0           # Hip abduction angle in degrees 
force = 17                             # Force applied at ankle in kg
ankle_force_N = 9.8 * force            # Force applied at ankle in Newtons
limb_length_m = 0.8                    # Distance from hip joint to ankle in meters

# ------------------------------------
# CALCULATE REQUIRED HIP TORQUE
# ------------------------------------
hip_torque_required = ankle_force_N * limb_length_m  # Nm

# ------------------------------------
# MUSCLE DEFINITIONS (simplified)
# ------------------------------------
muscles = {
    "glute_med": {"Fmax": 2000, "moment_arm": limb_length_m*0.01*(5.5 + 0.04*hip_abduction_angle_deg + 0.0002*hip_abduction_angle_deg*hip_abduction_angle_deg)},   # meters
    "glute_min": {"Fmax": 1700, "moment_arm": limb_length_m*0.01*(4.5 + 0.04*hip_abduction_angle_deg + 0.0002*hip_abduction_angle_deg*hip_abduction_angle_deg)},
    "TFL":       {"Fmax": 500,  "moment_arm": limb_length_m*0.01*(-1+ 1.1*math.log(hip_abduction_angle_deg+30))}
}

muscle_names = list(muscles.keys())
Fmax = np.array([muscles[m]["Fmax"] for m in muscle_names])
moment_arms = np.array([muscles[m]["moment_arm"] for m in muscle_names])

# ------------------------------------
# OPTIMIZATION SETUP
# ------------------------------------
def objective(F):
    """Minimize sum of squared normalized forces (activation cost)."""
    return np.sum((F / Fmax) ** 2)

def torque_constraint(F):
    """Ensure total torque generated equals required hip torque."""
    return np.dot(F, moment_arms) - hip_torque_required

# Constraints and bounds
constraints = {'type': 'eq', 'fun': torque_constraint}
bounds = [(0, fmax) for fmax in Fmax]
initial_guess = np.zeros(len(Fmax))

# ------------------------------------
# RUN OPTIMIZATION
# ------------------------------------
result = minimize(
    fun=objective,
    x0=initial_guess,
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'
)

# ------------------------------------
# OUTPUT RESULTS
# ------------------------------------
print(f"Required hip abduction torque: {hip_torque_required:.2f} Nm\n")
glute_med = muscles['glute_med']['moment_arm']
glute_min = muscles['glute_min']['moment_arm']
TFL = muscles['TFL']['moment_arm']
print(f"Moment Arms:\n  Glute Med: {glute_med:.2f} m\n  Glute Min: {glute_min:.2f} m\n  TFL: {TFL:.2f} m")

if result.success:
    print("Estimated Muscle Contributions:")
    for i, muscle in enumerate(muscle_names):
        force = result.x[i]
        print(f"  {muscle:<12} : {force:.2f} N")
else:
    print("⚠️ Optimization failed:", result.message)
