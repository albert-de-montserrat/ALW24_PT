```@meta
EditURL = "../scripts/4_Linear_PT_accelerated.jl"
```

# The Accelerated Pseudo-Transient iterative method

We can improve the convergence rate of the previous PT solver by also doing a continuation also in the flux equation:

$$\psi \frac{\partial q_x}{\partial \tau}  + q_x = -K\frac{\partial h}{\partial x}$$
$$\psi \frac{\partial q_y}{\partial \tau}  + q_y = -K\frac{\partial h}{\partial y}$$

## Discretisation
We can solve the PT equations iteratively by simply doing explicit updates. Therefore, the discretised equations looks like

$$\psi \frac{q_{(x)i+\frac{1}{2}, j}^{n+1} - q_{(x)i+\frac{1}{2}, j}^n}{\Delta \tau} + q_{(x)i+\frac{1}{2}, j}^n = -K \frac{h_{i+1,j}^n - h_{i,j}^n}{\Delta x}$$

# Further accelerating the PT iterations

So far we have discretised both the flux and diffusion PT equations in an explicit manner. We can further improve the convergence of the iterative solver by discretising these equations in an implicit way.

$$\psi \frac{q_{(x)i+\frac{1}{2}, j}^{n+1} - q_{(x)i+\frac{1}{2}, j}^n}{\Delta \tau} + q_{(x)i+\frac{1}{2}, j}^{n+1} = -K \frac{h_{i+1,j}^n - h_{i,j}^n}{\Delta x}$$

$$\psi \frac{q_{(x)i, j+\frac{1}{2}}^{n+1} - q_{(x)i, j+\frac{1}{2}}^n}{\Delta \tau} + q_{(x)i, j+\frac{1}{2}}^{n+1} = -K \frac{h_{i,j+1}^n - h_{i,j}^n}{\Delta x}$$

and

$$\theta \frac{h^{n+1}_i - h^n_i}{\Delta \tau} + \frac{h^{n+1}_i - h^t_i}{\Delta t} = -\left(\frac{ q^{n+1}_{(x)i+\frac{1}{2},j} - q^{n+1}_{(x)-\frac{1}{2},j} }{\Delta x} +\frac{q^{n+1}_{(y)i,j+\frac{1}{2}} - q^{n+1}_{(y)i,j-\frac{1}{2}}}{\Delta x}\right)$$

# Time to pogram!

Modify your solver to use the accelerated PT diffusion equation.

## PT coefficients

We now have the additional $\psi$ damping coefficient. Use:
$$ \frac{\alpha}{\Delta\tau} = \frac{L}{V_p \Delta\tau Re}$$

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

