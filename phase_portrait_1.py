# ##############################################################################
# Author: Edward E. Daisey
# Course: Modeling & Simulation of Complex Systems
# Date: 10th of March, 2026
# Title: Problem 8(d) - Phase Portrait for a Two-Dimensional Nonlinear System
###############################################################################

################################### Overview ##################################
# Description:
#   This script generates a phase portrait for the system
#
#       dx/dt = -2y - epsilon*x^17
#       dy/dt =  8x - epsilon*y^13
#
#   with epsilon = 1/1000 on the window
#
#       |x| <= 4,   |y| <= 4.
#
# Problem:
#   Problem 8(d) asks for a computer-generated phase portrait for this system.
# ##############################################################################


# ############################## Imports #######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# ##############################################################################


# ############################## Constants #####################################
epsilon = 1.0 / 1000.0

xMin = -4.0
xMax =  4.0
yMin = -4.0
yMax =  4.0

timeStart = 0.0
timeEnd = 40.0

relativeTolerance = 1.0e-8
absoluteTolerance = 1.0e-10

gridSize = 29
# ##############################################################################


# ############################## Function 1 ####################################
# Name:
#   VectorField
#
# Purpose:
#   Returns the right-hand side of the system
#
#       dx/dt = -2y - epsilon*x^17
#       dy/dt =  8x - epsilon*y^13
#
# Input:
#   t     : Time variable (included for solve_ivp compatibility).
#   state : Current state vector [x, y].
#
# Output:
#   A list [dx, dy].
def VectorField(t, state):
    x, y = state

    dx = -2.0 * y - epsilon * x**17
    dy =  8.0 * x - epsilon * y**13

    return [dx, dy]
# ##############################################################################


# ############################## Function 2 ####################################
# Name:
#   GenerateInitialConditions
#
# Purpose:
#   Creates a collection of initial conditions on the boundary of the square
#   |x| <= 4, |y| <= 4 so that the phase portrait shows many trajectories.
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

    # A few interior points near the origin:
    initialConditions.append([ 1.0,  0.0])
    initialConditions.append([-1.0,  0.0])
    initialConditions.append([ 0.0,  1.0])
    initialConditions.append([ 0.0, -1.0])
    initialConditions.append([ 0.75,  0.75])
    initialConditions.append([-0.75, -0.75])

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
        dense_output=False,
        max_step=0.05
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

    U = -2.0 * Y - epsilon * X**17
    V =  8.0 * X - epsilon * Y**13

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

    plt.plot(0.0, 0.0, marker="o", markersize=6)

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Problem 8(d): Phase Portrait for $\epsilon = 1/1000$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
# ##############################################################################


# ############################## Function 6 ####################################
# Name:
#   Main
#
# Purpose:
#   Produces the phase portrait requested in Problem 8(d).
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