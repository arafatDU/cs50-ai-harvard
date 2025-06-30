'''
Linear Programming Example: Production Optimization

Two machine X1 and X2. X1 cost $50 per hour to run, X2 cost $80 per hour to run. Goal is to minimize the cost
X1 requires 5 unit labor per hour, X2 requires 2 unit labor per hour. Total labor available is 20 units.
X1 produces 10 units per hour, X2 produces 12 units per hour. Goal is to produce at least 90 units.
'''




import scipy.optimize

# Objective Function: 50x_1 + 80x_2
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90

result = scipy.optimize.linprog(
    [50, 80],  # Cost function: 50x_1 + 80x_2
    A_ub=[[5, 2], [-10, -12]],  # Coefficients for inequalities
    b_ub=[20, -90],  # Constraints for inequalities: 20 and -90
)

if result.success:
    print(f"X1: {round(result.x[0], 2)} hours")
    print(f"X2: {round(result.x[1], 2)} hours")
else:
    print("No solution")