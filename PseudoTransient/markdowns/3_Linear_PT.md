```@meta
EditURL = "../scripts/3_Linear_PT.jl"
```

# The Pseudo-Transient iterative method: hillslope diffusion
Let's start by refreshing the hillslope diffusion equation from the previous notebook:

$$q_x = -K\frac{\partial h}{\partial x}$$
$$q_y = -K\frac{\partial h}{\partial y}$$

$$\frac{\partial h}{\partial t} = -\left(\frac{\partial q_x}{\partial x} + \frac{\partial q_y}{\partial y} \right)$$

Rearranging the previous equation we can define the residual

$$R =-\left(\frac{\partial q_x}{\partial x} + \frac{\partial q_y}{\partial y} \right) -  \frac{\partial h}{\partial t} = 0$$

The Pseudo-Transient (PT) method is an iterative approach that augments the PDEs with a pseudo-time derivative $\partial/\partial\tau$ and let it reach a steady state:

$$\theta \frac{\partial h}{\partial \tau} = R$$

where $\theta$ is a constant and $\tau$ is usually referred as the pseudo-timestep. Therefore we can see that the new term $\theta \frac{\partial h}{\partial \tau}$ should be zero and vanish upon convergence.

## Discretisation
We can solve the PT equations iteratively by simply doing explicit updates. Therefore the discretised equations look like

$$q^{(x)}_{i+\frac{1}{2},j} = -K \frac{h_{i+1,j}^n - h_{i,j}^n}{\Delta x}$$
$$q^{(y)}_{i,j+\frac{1}{2}} = -K \frac{h_{i,j+1}^n - h_{i,j}^n}{\Delta y}$$

$$\theta \frac{h^{n+1}_i - h^n_i}{\Delta \tau} = \left(\frac{ q^{(x)}_{i+\frac{1}{2},j} - q^{(x)}_{i-\frac{1}{2},j} }{\Delta x} +\frac{q^{(y)}_{i,j+\frac{1}{2}} - q^{(y)}_{i,j-\frac{1}{2}}}{\Delta x}\right) - \frac{h^n_i - h^t_i}{\Delta t} $$

where the superscript $n$ is the PT iteration counter. As you can see the flux equation remains the same as in the explicit solver and we only need to slightly modify the $h$ update function to accommodate for the new terms of the PT equation.

# 💾 Time to program 💾

## Arrays

we will need an additional `Array` to solve the PT equations:

|     Array      |        size      |
|:--------------:|:----------------:|
| $\mathbf{h}$   | $n   \times n  $ |
| $\mathbf{q}_x$ | $n-1 \times n-2$ |
| $\mathbf{q}_y$ | $n-2 \times n-1$ |
| $\mathbf{R}$   | $n-2 \times n-2$ |

## PT coefficients

We now have to additional parameters in the equations, namely $\theta$ and $\tau$, that need to be tuned to obtain an optimal convergence rate. Use the following values from [(Räss et al., 2022)](https://doi.org/10.5194/gmd-15-5757-2022):

$$C \leq 1$$

$$V_p\Delta \tau = C \Delta x$$

$$Re = π + \sqrt{π^2 + \frac{L^2}{K \Delta t}}$$

$$\frac{\Delta\tau}{\theta} = \frac{V_p \Delta \tau L}{K Re}$$

## Exercises

1. Modify the previous explicit solver to solve the PT equations. You will need to do the following modifications:
    1. Compute the resisudal $R$
    2. New `update_h` function
    3. Introduce the PT iterations within the time stepping loop. Should look similar to:
```julia
while t < run_time
    ϵ    = 1e-9 # tolerance
    er   = Inf  # global error
    iter = 0    # PT iteration counter
    while er > ϵ
        # update iteration counter
        iter += 1
        # Compute flux (as we did in the previous exercise)
        qx .= ??
        qy .= ??
        # Compute the residual R
        R .= ??
        # Update h
        @views h[2:end-1, 2:end-1,] .= ??
        # check global error
        er = norm(R) / n^2
    end
    println("Convergence reached in $iter iterations with error $er")
    copyto!(h0, h)
    nt += 1
    t  += dt
end
```
2. How far can we push $\Delta t$ now?

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

