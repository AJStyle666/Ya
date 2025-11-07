# gradient_descent_example.py
import numpy as np

def f(x):
    return (x + 3)**2

def grad(x):
    return 2 * (x + 3)

# Gradient descent
x = 2.0  # start
lr = 0.1  # learning rate
n_iter = 100
history = []

for i in range(n_iter):
    g = grad(x)
    x = x - lr * g
    history.append((i, x, f(x)))
    # optional stopping
    if abs(g) < 1e-6:
        break

print(f"After {len(history)} steps, x ≈ {x:.6f}, f(x) ≈ {f(x):.6f}")
# Expected minima at x = -3, f(-3) = 0
# Print first few steps for clarity
print("First 5 steps:")
for t in history[:5]:
    print(t)
