import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import make_interp_spline
import seaborn as sns

import Fit_functions as fit

plt.rcParams.update({'font.size': 13})
df_results = pd.DataFrame()
method = 'Grover'

trials = 10
Ideal_value_trials = []
scale = []
gate_left_trials, gate_right_trials, gate_random_trials, circuit_trials= [], [], [], []

for trial in range(trials):
    df = pd.read_csv('Results\Grover\Results_Grover_4_{}.csv '.format(trial))
    Ideal_value_trials.append(df['Ideal'][0])
    scale.append(df['Scale'].to_numpy())
    gate_left_trials.append(df['Gate Left'].to_numpy())
    gate_right_trials.append(df['Gate Right'].to_numpy())
    gate_random_trials.append(df['Gate Random'].to_numpy())
    circuit_trials.append(df['Circuit'].to_numpy())

Ideal_value = np.mean(Ideal_value_trials)
sigma_ideal = np.std(Ideal_value_trials)
scale = np.mean(scale, axis = 0)

gate_left = np.mean(gate_left_trials, axis=0)
sigma_gate_left = np.std(gate_left_trials, axis=0)

gate_right = np.mean(gate_right_trials, axis=0)
sigma_gate_right = np.std(gate_right_trials, axis=0)

gate_random = np.mean(gate_random_trials, axis=0)
sigma_gate_random = np.std(gate_random_trials, axis=0)

circuit = np.mean(circuit_trials, axis=0)
sigma_circuit = np.std(circuit_trials, axis=0)

#performing the fit of the circuit data
print('Circuit')
unmitigated_error = np.abs(Ideal_value-circuit[0])/Ideal_value
print('Unmitigated error: ', unmitigated_error ,'+/-', unmitigated_error* np.sqrt((sigma_circuit[0]/circuit[0])**2 + (sigma_ideal/Ideal_value)**2))
#Linear fit
p_0 = [0.05, 1]
params, cov = opt.curve_fit(fit.linear, scale, circuit, p0=p_0, sigma=sigma_circuit, absolute_sigma=True)
circuit_opt_linear = fit.linear(scale, *params)
extrapolated_linear = fit.linear(0, *params)
err_extrapolated_linear = np.sqrt(cov[1,1])

err_circuit_lin = np.abs(Ideal_value-extrapolated_linear)/Ideal_value
sigma_err_circuit_lin = err_circuit_lin * np.sqrt((err_extrapolated_linear/extrapolated_linear)**2 + (sigma_ideal/Ideal_value)**2)
print('Linear: ', err_circuit_lin, '+/-', sigma_err_circuit_lin)

#Exponential fit
p_0 = [0.4, 0.1, 2]
params, cov = opt.curve_fit(fit.exponential, scale, circuit, p0=p_0, sigma=sigma_circuit, absolute_sigma=True, maxfev=2000)
scale_new = np.linspace(scale.min(), scale.max(), 300)
spl = make_interp_spline(scale, fit.exponential(scale, *params), k=3)
circuit_opt_exponential = spl(scale_new)
extrapolated_exponential = fit.exponential(0, *params)
err_extrapolated_exponential = extrapolated_exponential * np.sqrt(cov[0,0]/params[0]**2 + cov[1,1]/params[1]**2)

err_circuit_exp = np.abs(Ideal_value-extrapolated_exponential)/Ideal_value
sigma_err_circuit_exp = err_circuit_exp * np.sqrt((err_extrapolated_exponential/extrapolated_exponential)**2 + (sigma_ideal/Ideal_value)**2)
print(err_circuit_exp, '+/-', sigma_err_circuit_exp)

#Poly 2 fit
p_0 = [1, -50, 0.98]
params, cov = opt.curve_fit(fit.poly_order_2, scale, circuit, p0=p_0, sigma=sigma_circuit, absolute_sigma=True, maxfev=2000)
spl = make_interp_spline(scale, fit.poly_order_2(scale, *params), k=3) #k is B-spline degrees, default is cubic
circuit_opt_poly2 = spl(scale_new)
extrapolated_poly2 = fit.poly_order_2(0, *params)
err_extrapolated_poly2 = np.sqrt(cov[2,2])

err_circuit_pol2 = np.abs(Ideal_value-extrapolated_poly2)/Ideal_value
sigma_err_circuit_pol2 = err_circuit_pol2 * np.sqrt((err_extrapolated_poly2/extrapolated_poly2)**2 + (sigma_ideal/Ideal_value)**2)
print('poly: ', err_circuit_pol2, '+/-', sigma_err_circuit_pol2)
sns.set(font_scale=1.7)
fig, ax = plt.subplots(figsize=(10,7))
#plotting points and correct value
plt.errorbar(scale, circuit, yerr=sigma_circuit, color='blue',label='Sample', fmt='o')
plt.axhline(y=Ideal_value, color='r', linestyle='-', label= 'Ideal Value')

#plotting fit functions
plt.plot(scale, circuit_opt_linear, label= "Fit Linear", color='orange')
plt.plot(scale_new, circuit_opt_poly2, label= "Fit Poly 2", color='magenta')
plt.plot(scale_new, circuit_opt_exponential, label= "Fit Exponential", color='green')


#plotting extrapolated points
plt.errorbar(0, extrapolated_linear, yerr=err_extrapolated_linear, fmt='v', color = 'orange', label ='Linear')
plt.errorbar(0, extrapolated_exponential, yerr=err_extrapolated_exponential, fmt='s',color='green' ,label = 'Exp')
plt.errorbar(0, extrapolated_poly2, yerr=err_extrapolated_poly2, fmt='*', color = 'magenta', label = 'Poly 2')

plt.xlim([-0.5, scale[-1]+2])
ax.set_title("Circuit Folding")
ax.set_xlabel("Scale")
ax.set_ylabel("Value")
plt.legend()

plt.show()

###############################################
#performing fit for the Left Gate circuit
print('Left Gate')
unmitigated_error = np.abs(Ideal_value-gate_left[0])/Ideal_value
print('Unmitigated error: ', unmitigated_error ,'+/-', unmitigated_error * np.sqrt((sigma_gate_left[0]/gate_left[0])**2 + (sigma_ideal/Ideal_value)**2))

#Linear fit
p_0 = [0.05, 1]
params, cov = opt.curve_fit(fit.linear, scale, gate_left, p0=p_0, sigma=sigma_gate_left, absolute_sigma=True)
circuit_opt_linear = fit.linear(scale, *params)
extrapolated_linear = fit.linear(0, *params)
err_extrapolated_linear = np.sqrt(cov[1,1])

err_left_lin = np.abs(Ideal_value-extrapolated_linear)/Ideal_value
sigma_err_left_lin = err_left_lin * np.sqrt((err_extrapolated_linear/extrapolated_linear)**2 + (sigma_ideal/Ideal_value)**2)

#Exponential fit
p_0 = [0.4, 0.1, 2]
params, cov = opt.curve_fit(fit.exponential, scale, gate_left, p0=p_0, sigma=sigma_gate_left, absolute_sigma=True, maxfev=2000)
scale_new = np.linspace(scale.min(), scale.max(), 300)
spl = make_interp_spline(scale, fit.exponential(scale, *params), k=3)
circuit_opt_exponential = spl(scale_new)
extrapolated_exponential = fit.exponential(0, *params)
err_extrapolated_exponential = extrapolated_exponential * np.sqrt(cov[0,0]/params[0]**2 + cov[1,1]/params[1]**2)

err_left_exp = np.abs(Ideal_value-extrapolated_exponential)/Ideal_value
sigma_err_left_exp = err_left_exp * np.sqrt((err_extrapolated_exponential/extrapolated_exponential)**2 + (sigma_ideal/Ideal_value)**2)
print(err_left_exp, '+/-', sigma_err_left_exp)

#Poly 2 fit
p_0 = [1, -50, 0.98]
params, cov = opt.curve_fit(fit.poly_order_2, scale, gate_left, p0=p_0, sigma=sigma_gate_left, absolute_sigma=True, maxfev=2000)
spl = make_interp_spline(scale, fit.poly_order_2(scale, *params), k=3)
circuit_opt_poly2 = spl(scale_new)
extrapolated_poly2 = fit.poly_order_2(0, *params)
err_extrapolated_poly2 = np.sqrt(cov[2,2])

err_left_pol2 = np.abs(Ideal_value-extrapolated_poly2)/Ideal_value
sigma_err_left_pol2 = err_left_pol2 * np.sqrt((err_extrapolated_poly2/extrapolated_poly2)**2 + (sigma_ideal/Ideal_value)**2)

fig, ax = plt.subplots(figsize=(10,7))
#plotting points and correct value
plt.errorbar(scale, gate_left, yerr=sigma_gate_left, color='blue',label='Sample', fmt='o')
plt.axhline(y=Ideal_value, color='r', linestyle='-', label= 'Ideal Value')

#plotting fit functions
plt.plot(scale, circuit_opt_linear, label= "Fit Linear", color='orange')
plt.plot(scale_new, circuit_opt_poly2, label= "Fit Poly 2", color='magenta')
plt.plot(scale_new, circuit_opt_exponential, label= "Fit Exponential", color='green')

#plotting extrapolated points
plt.errorbar(0, extrapolated_linear, yerr=err_extrapolated_linear, fmt='v', color = 'orange', label ='Linear')
plt.errorbar(0, extrapolated_poly2, yerr=err_extrapolated_poly2, fmt='*', color = 'magenta', label = 'Poly 2')
plt.errorbar(0, extrapolated_exponential, yerr=err_extrapolated_exponential,  fmt='s',color='green' ,label = 'Exp')

plt.xlim([-0.5, scale[-1]+2])
ax.set_title("Gate Folding, Left")
ax.set_xlabel("Scale")
ax.set_ylabel("Value")
plt.legend()

plt.show()

###############################################
#performing fit for the Right Gate circuit
print('Right Gate')
unmitigated_error = np.abs(Ideal_value-gate_right[0])/Ideal_value
print('Unmitigated error: ', unmitigated_error ,'+/-', unmitigated_error * np.sqrt((sigma_gate_right[0]/gate_right[0])**2 + (sigma_ideal/Ideal_value)**2))

#Linear fit
p_0 = [0.05, 1]
params, cov = opt.curve_fit(fit.linear, scale, gate_right, p0=p_0, sigma=sigma_gate_right, absolute_sigma=True)
circuit_opt_linear = fit.linear(scale, *params)
extrapolated_linear = fit.linear(0, *params)
err_extrapolated_linear = np.sqrt(cov[1,1])

err_right_lin = np.abs(Ideal_value-extrapolated_linear)/Ideal_value
sigma_err_right_lin = err_right_lin * np.sqrt((err_extrapolated_linear/extrapolated_linear)**2 + (sigma_ideal/Ideal_value)**2)


#Exponential fit
p_0 = [0.4, 0.1, 2]
params, cov = opt.curve_fit(fit.exponential, scale, gate_right, p0=p_0, sigma=sigma_gate_right, absolute_sigma=True, maxfev=2000)
scale_new = np.linspace(scale.min(), scale.max(), 300)
spl = make_interp_spline(scale, fit.exponential(scale, *params), k=3)
circuit_opt_exponential = spl(scale_new)
extrapolated_exponential = fit.exponential(0, *params)
err_extrapolated_exponential = extrapolated_exponential * np.sqrt(cov[0,0]/params[0]**2 + cov[1,1]/params[1]**2)

err_right_exp = np.abs(Ideal_value-extrapolated_exponential)/Ideal_value
sigma_err_right_exp = err_right_exp * np.sqrt((err_extrapolated_exponential/extrapolated_exponential)**2 + (sigma_ideal/Ideal_value)**2)

print(err_right_exp, '+/-', sigma_err_right_exp)


#Poly 2 fit
p_0 = [1, -50, 0.98]
params, cov = opt.curve_fit(fit.poly_order_2, scale, gate_right, p0=p_0, sigma=sigma_gate_right, absolute_sigma=True, maxfev=2000)
spl = make_interp_spline(scale, fit.poly_order_2(scale, *params), k=3) #k is B-spline degrees, default is cubic
circuit_opt_poly2 = spl(scale_new)
extrapolated_poly2 = fit.poly_order_2(0, *params)
err_extrapolated_poly2 = np.sqrt(cov[2,2])

err_right_pol2 = np.abs(Ideal_value-extrapolated_poly2)/Ideal_value
sigma_err_right_pol2 = err_right_pol2 * np.sqrt((err_extrapolated_poly2/extrapolated_poly2)**2 + (sigma_ideal/Ideal_value)**2)

fig, ax = plt.subplots(figsize=(10,7))
#plotting points and correct value
plt.errorbar(scale, gate_right, yerr=sigma_gate_right, color='blue',label='Sample', fmt='o')
plt.axhline(y=Ideal_value, color='r', linestyle='-', label= 'Ideal Value')

#plotting fit functions
plt.plot(scale, circuit_opt_linear, label= "Fit Linear", color='orange')
plt.plot(scale_new, circuit_opt_poly2, label= "Fit Poly 2", color='magenta')
plt.plot(scale_new, circuit_opt_exponential, label= "Fit Exponential", color='green')


#plotting extrapolated points
plt.errorbar(0, extrapolated_linear, yerr=err_extrapolated_linear, fmt='v', color = 'orange', label ='Linear')
plt.errorbar(0, extrapolated_poly2, yerr=err_extrapolated_poly2, fmt='*', color = 'magenta', label = 'Poly 2')
plt.errorbar(0, extrapolated_exponential,yerr=err_extrapolated_exponential,  fmt='s',color='green' ,label = 'Exp')


plt.xlim([-0.5, scale[-1]+2])
ax.set_title("Gate Folding, Right")
ax.set_xlabel("Scale")
ax.set_ylabel("Value")
plt.legend()

plt.show()


###############################################
#performing fit for the Random Gate circuit
print('Random Gate')
unmitigated_error = np.abs(Ideal_value-gate_random[0])/Ideal_value
print('Unmitigated error: ', unmitigated_error ,'+/-', unmitigated_error * np.sqrt((sigma_gate_random[0]/gate_random[0])**2 + (sigma_ideal/Ideal_value)**2))
#Linear fit
p_0 = [0.05, 1]
params, cov = opt.curve_fit(fit.linear, scale, gate_random, p0=p_0, sigma=sigma_gate_random, absolute_sigma=True)
circuit_opt_linear = fit.linear(scale, *params)
extrapolated_linear = fit.linear(0, *params)
err_extrapolated_linear = np.sqrt(cov[1,1])

err_random_lin = np.abs(Ideal_value-extrapolated_linear)/Ideal_value
sigma_err_random_lin = err_random_lin * np.sqrt((err_extrapolated_linear/extrapolated_linear)**2 + (sigma_ideal/Ideal_value)**2)


#Exponential fit
p_0 = [0.4, 0.1, 2]
params, cov = opt.curve_fit(fit.exponential, scale, gate_random, p0=p_0, sigma=sigma_gate_random, absolute_sigma=True, maxfev=2000)
scale_new = np.linspace(scale.min(), scale.max(), 300)
spl = make_interp_spline(scale, fit.exponential(scale, *params), k=3)
circuit_opt_exponential = spl(scale_new)
extrapolated_exponential = fit.exponential(0, *params)
err_extrapolated_exponential = extrapolated_exponential * np.sqrt(cov[0,0]/params[0]**2 + cov[1,1]/params[1]**2)

err_random_exp = np.abs(Ideal_value-extrapolated_exponential)/Ideal_value
sigma_err_random_exp = err_random_exp * np.sqrt((err_extrapolated_exponential/extrapolated_exponential)**2 + (sigma_ideal/Ideal_value)**2)

print(err_random_exp, '+/-', sigma_err_random_exp)

#Poly 2 fit
p_0 = [1, -50, 0.98]
params, cov = opt.curve_fit(fit.poly_order_2, scale, gate_random, p0=p_0, sigma=sigma_gate_random, absolute_sigma=True, maxfev=2000)
spl = make_interp_spline(scale, fit.poly_order_2(scale, *params), k=3) #k is B-spline degrees, default is cubic
circuit_opt_poly2 = spl(scale_new)
extrapolated_poly2 = fit.poly_order_2(0, *params)
err_extrapolated_poly2 = np.sqrt(cov[2,2])

err_random_pol2 = np.abs(Ideal_value-extrapolated_poly2)/Ideal_value
sigma_err_random_pol2 = err_random_pol2 * np.sqrt((err_extrapolated_poly2/extrapolated_poly2)**2 + (sigma_ideal/Ideal_value)**2)


fig, ax = plt.subplots(figsize=(10,7))
#plotting points and correct value
plt.errorbar(scale, gate_random, yerr=sigma_gate_random, color='blue',label='Sample', fmt='o')
plt.axhline(y=Ideal_value, color='r', linestyle='-', label= 'Ideal Value')

#plotting fit functions
plt.plot(scale, circuit_opt_linear, label= "Fit Linear", color='orange')
plt.plot(scale_new, circuit_opt_poly2, label= "Fit Poly 2", color='magenta')
plt.plot(scale_new, circuit_opt_exponential, label= "Fit Exponential", color='green')

#plotting extrapolated points
plt.errorbar(0, extrapolated_linear, yerr=err_extrapolated_linear, fmt='v', color = 'orange', label ='Linear')
plt.errorbar(0, extrapolated_exponential, yerr=err_extrapolated_exponential,  fmt='s',color='green' ,label = 'Exp')
plt.errorbar(0, extrapolated_poly2, yerr=err_extrapolated_poly2, fmt='*', color = 'magenta', label = 'Poly 2')

plt.xlim([-0.5, scale[-1]+2])
ax.set_title("Gate Folding, Random")
ax.set_xlabel("Scale")
ax.set_ylabel("Value")
plt.legend()

plt.show()

#saving results in a csv file
new_row = pd.Series(data={"A_Ideal": Ideal_value,
                          "Circuit exp": err_circuit_exp*100, "Circuit exp sigma": sigma_err_circuit_exp*100,
                          "Circuit lin": err_circuit_lin*100, "Circuit lin sigma":sigma_err_circuit_lin*100,
                          "Circuit pol2": err_circuit_pol2*100, "Circuit pol2 sigma":sigma_err_circuit_pol2*100,
                          "Left exp": err_left_exp*100, "Left exp sigma": sigma_err_left_exp*100,
                          "Left lin": err_left_lin*100, "Left lin sigma": sigma_err_left_lin*100,
                          "Left pol2": err_left_pol2*100, "Left pol2 sigma": sigma_err_left_pol2*100,
                          "Right exp": err_right_exp*100, "Right exp sigma": sigma_err_right_exp*100,
                          "Right lin": err_right_lin*100, "Right lin sigma": sigma_err_right_lin*100,
                          "Right pol2": err_right_pol2*100, "Right pol2 sigma": sigma_err_right_pol2*100,
                          "Random exp": err_random_exp*100, "Random exp sigma": sigma_err_random_exp*100,
                          "Random lin": err_random_lin*100, "Random lin sigma": sigma_err_random_lin*100,
                          "Random pol2": err_random_pol2*100, "Random pol2 sigma": sigma_err_random_pol2*100},
                          name='{}'.format(method))

df_results = df_results.append(new_row, ignore_index=False)
#df_results.to_csv('Results\Errors.csv', mode='a', index=True)
