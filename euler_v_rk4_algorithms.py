# ##############################################################################
# Author: Edward E. Daisey
# Course: Modeling & Simulation of Complex Systems
# Date: 10th of March, 2026
# Title: Problem 9 - Euler and RK4 Approximation for x(2)
###############################################################################

################################### Overview ##################################
# Description:
#   This script approximates x(2) for the initial-value problem
#
#       dx/dt = x - x^4
#       x(0)  = 1/10
#
#   using:
#       (a) Euler's Method with Delta t = 1/64
#       (b) RK4 Method with Delta t = 1/16
#
# Problem:
#   Find the approximate value for x(2), displaying at least six digits
#   of precision.
#
# Initial Condition:
#   x(0) = 0.1
#
# Step Sizes:
#   Euler: Delta t = 1/64
#   RK4  : Delta t = 1/16
#
# Time Interval:
#   t in [0, 2]
#
# Reproducibility:
#   (1) Install the required Python packages.
#   (2) Run this script.
#   (3) The script will print the Euler and RK4 approximations for x(2).
#
# Reference:
#   Problem 9 from the attached Modeling & Simulation of Complex Systems
#   midterm/problem set.
# ##############################################################################


# ############################## Imports #######################################
# No third-party packages are required for these calculations.
# ##############################################################################


# ############################## Constants #####################################
initialValue = 1.0 / 10.0

eulerStepSize = 1.0 / 64.0
rk4StepSize = 1.0 / 16.0

timeStart = 0.0
timeEnd = 2.0
# ##############################################################################


# ############################## Function 1 ####################################
# Name:
#   VectorField
#
# Purpose:
#   Returns the right-hand side of the differential equation
#
#       dx/dt = x - x^4
#
# Input:
#   x : Current value of the state variable.
#
# Output:
#   The value of dx/dt.
def VectorField(x):
    dx = x - x**4
    return dx
# ##############################################################################


# ############################## Function 2 ####################################
# Name:
#   EulerMethod
#
# Purpose:
#   Approximates x(2) using Euler's Method with Delta t = 1/64.
#
# Input:
#   None.
#
# Output:
#   The Euler approximation for x(2).
def EulerMethod():
    currentTime = timeStart
    currentValue = initialValue

    numberOfSteps = int((timeEnd - timeStart) / eulerStepSize)

    for stepIndex in range(numberOfSteps):
        currentValue = currentValue + eulerStepSize * VectorField(currentValue)
        currentTime = currentTime + eulerStepSize

    return currentValue
# ##############################################################################


# ############################## Function 3 ####################################
# Name:
#   RK4Method
#
# Purpose:
#   Approximates x(2) using the classical fourth-order Runge-Kutta method
#   with Delta t = 1/16.
#
# Input:
#   None.
#
# Output:
#   The RK4 approximation for x(2).
def RK4Method():
    currentTime = timeStart
    currentValue = initialValue

    numberOfSteps = int((timeEnd - timeStart) / rk4StepSize)

    for stepIndex in range(numberOfSteps):
        k1 = VectorField(currentValue) * rk4StepSize
        k2 = VectorField(currentValue + 0.5 * k1) * rk4StepSize
        k3 = VectorField(currentValue + 0.5 * k2) * rk4StepSize
        k4 = VectorField(currentValue + k3) * rk4StepSize

        currentValue = currentValue + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        currentTime = currentTime + rk4StepSize

    return currentValue
# ##############################################################################


# ############################## Function 4 ####################################
# Name:
#   Main
#
# Purpose:
#   Computes and prints the Euler and RK4 approximations for x(2).
#
# Input:
#   None.
#
# Output:
#   Prints the requested approximations.
def Main():
    eulerApproximation = EulerMethod()
    rk4Approximation = RK4Method()

    print("Problem 9 Results")
    print("-----------------")
    print(f"Euler Method (Delta t = 1/64): x(2) ≈ {eulerApproximation:.12f}")
    print(f"RK4 Method   (Delta t = 1/16): x(2) ≈ {rk4Approximation:.12f}")
# ##############################################################################


# ############################### Execution ####################################
Main()
# ##############################################################################