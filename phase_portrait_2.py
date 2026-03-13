# ##############################################################################
# Author: Edward E. Daisey
# Course: Modeling & Simulation of Complex Systems
# Date: 10th of March, 2026
# Title: Problem 10(c) - Phase Portrait for a Two-Dimensional Nonlinear System
###############################################################################

################################### Overview ##################################
# Description:
#   This script generates the phase portrait requested in Problem 10(c) for the
#   two-dimensional system
#
#       dx/dt = cos(x/2) * (mu*x - omega*y)
#       dy/dt = cos(y/2) * (omega*x + mu*y)
#
# Problem:
#   Use a computer to generate a phase portrait with mu = 1/4, omega = 1.
#   For the plot use the window |x| <= 4, |y| <= 4.
#
# Parameters:
#   mu    = 1/4
#   omega = 1
#
# Plotting Window:
#   x in [-4, 4]
#   y in [-4, 4]
#
# Reproducibility:
#   (1) Install the required Python packages.
#   (2) Run this script.
#   (3) The script will display the requested phase portrait.
#
# Reference:
#   Problem 10(c) from the attached Modeling & Simulation of Complex Systems
#   midterm/problem set.
# ##############################################################################


# ############################## Imports #######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# ##############################################################################


# ############################## Constants #####################################
mu = 1.0 / 4.0
omega = 1.0

xMin = -4.0
xMax =  4.0
yMin = -4.0
yMax =  4.0

timeStart = 0.0
timeEnd = 35.0

relativeTolerance = 1.0e-8
absoluteTolerance = 1.0e-10
maximumStep = 0.05

gridSize = 29
# ##############################################################################


# ############################## Function 1 ####################################
# Name:
#   VectorField
#
# Purpose:
#   Returns the right-hand side of the system
#
#       dx/dt = cos(x/2) * (mu*x - omega*y)
#       dy/dt = cos(y/2) * (omega*x + mu*y)
#
# Input:
#   t     : Time variable (included for solve_ivp compatibility).
#   state : Current state vector [x, y].
#
# Output:
#   A list [dx, dy].
def VectorField(t, state):
    x, y = state

    dx = np.cos(x / 2.0) * (mu * x - omega * y)
    dy = np.cos(y / 2.0) * (omega * x + mu * y)

    return [dx, dy]
# ##############################################################################


# ############################## Function 2 ####################################
# Name:
#   GenerateInitialConditions
#
# Purpose:
#   Creates a collection of initial conditions on the boundary and in the
#   interior of the square |x| <= 4, |y| <= 4.
#
# Input:
#   None.
#
# Output:
#   A list of initial condition pairs [x0, y0].
def GenerateInitialConditions():
    initialConditions = []

    samplePoints = np.linspace(-4.0, 4.0, 9)

    # Top and bottom edges:
    for x0 in samplePoints:
        initialConditions.append([x0,  4.0])
        initialConditions.append([x0, -4.0])

    # Left and right edges:
    for y0 in samplePoints:
        initialConditions.append([ 4.0, y0])
        initialConditions.append([-4.0, y0])

    # Interior points:
    interiorPoints = [
        [ 0.50,  0.00],
        [-0.50,  0.00],
        [ 0.00,  0.50],
        [ 0.00, -0.50],
        [ 1.00,  1.00],
        [-1.00,  1.00],
        [ 1.00, -1.00],
        [-1.00, -1.00],
        [ 2.00,  0.50],
        [-2.00, -0.50]
    ]

    for point in interiorPoints:
        initialConditions.append(point)

    return initialConditions
# ##############################################################################


# ############################## Function 3 ####################################
# Name:
#   SolveTrajectory
#
# Purpose:
#   Numerically solves the system from one initial condition.
#
# Input:
#   initialState : Initial condition [x0, y0].
#
# Output:
#   A SciPy solution object.
def SolveTrajectory(initialState):
    solution = solve_ivp(
        VectorField,
        (timeStart, timeEnd),
        initialState,
        rtol=relativeTolerance,
        atol=absoluteTolerance,
        max_step=maximumStep
    )

    return solution
# ##############################################################################


# ############################## Function 4 ####################################
# Name:
#   PlotVectorField
#
# Purpose:
#   Plots the normalized vector field on the square |x| <= 4, |y| <= 4.
#
# Input:
#   None.
#
# Output:
#   Adds the vector field to the current figure.
def PlotVectorField():
    xValues = np.linspace(xMin, xMax, gridSize)
    yValues = np.linspace(yMin, yMax, gridSize)
    X, Y = np.meshgrid(xValues, yValues)

    U = np.cos(X / 2.0) * (mu * X - omega * Y)
    V = np.cos(Y / 2.0) * (omega * X + mu * Y)

    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0.0] = 1.0

    U = U / magnitude
    V = V / magnitude

    plt.quiver(X, Y, U, V, angles="xy")
# ##############################################################################


# ############################## Function 5 ####################################
# Name:
#   PlotPhasePortrait
#
# Purpose:
#   Generates the phase portrait by plotting the normalized vector field and
#   several numerical trajectories.
#
# Input:
#   None.
#
# Output:
#   Displays the phase portrait.
def PlotPhasePortrait():
    plt.figure(figsize=(7, 7))

    PlotVectorField()

    initialConditions = GenerateInitialConditions()

    for initialState in initialConditions:
        solution = SolveTrajectory(initialState)
        plt.plot(solution.y[0], solution.y[1], linewidth=1.0)

    # Plot the five fixed points visible in the window:
    fixedPoints = np.array([
        [0.0, 0.0],
        [ np.pi,  np.pi],
        [ np.pi, -np.pi],
        [-np.pi,  np.pi],
        [-np.pi, -np.pi]
    ])
    plt.plot(fixedPoints[:, 0], fixedPoints[:, 1], marker="o", linestyle="None", markersize=6)

    # Plot the invariant nullcline square x = ±pi, y = ±pi:
    plt.axvline(np.pi, linewidth=1.5)
    plt.axvline(-np.pi, linewidth=1.5)
    plt.axhline(np.pi, linewidth=1.5)
    plt.axhline(-np.pi, linewidth=1.5)

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Problem 10(c): Phase Portrait for $\mu = 1/4$, $\omega = 1$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
# ##############################################################################


# ############################## Function 6 ####################################
# Name:
#   Main
#
# Purpose:
#   Produces the phase portrait requested in Problem 10(c).
#
# Input:
#   None.
#
# Output:
#   Displays the requested figure.
def Main():
    PlotPhasePortrait()
# ##############################################################################


# ############################### Execution ####################################
Main()
# ##############################################################################