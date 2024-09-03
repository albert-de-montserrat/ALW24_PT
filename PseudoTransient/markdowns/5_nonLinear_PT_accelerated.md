```@meta
EditURL = "../scripts/5_nonLinear_PT_accelerated.jl"
```

# The Accelerated Pseudo-Transient iterative method: non-linear diffusion

$$
\frac{\partial h}{\partial t} = -\nabla \cdot \left(K \nabla h \right)  \\
$$

$$
K = K_s \exp(-\lambda h_w)
$$

where $K_s$ is the submarine diffusion coefficient, $\lambda$ is the submarine decay constant, and the water depth

$$
h_w =
   \left\{\begin{array}{lr}
       h_s - h, & h_s \geq h \\
       0, & h_s < h
    \end{array}\right.
$$

where $h_s$ is the sea level elevation.

# 💾 Time to program 💾

## Arrays

$K$ now also has to be an `Array` since it is variable in space:

|     Array      |        size      |
|:--------------:|:----------------:|
| $K$            | $n   \times n  $ |
| $\mathbf{h}$   | $n   \times n  $ |
| $\mathbf{q}_x$ | $n-1 \times n-2$ |
| $\mathbf{q}_y$ | $n-2 \times n-1$ |
| $\mathbf{R}$   | $n-2 \times n-2$ |

Pseudo-Transient loop
Since now the PT coefficients depend in a non-linear K, they need to be recomputed in every iteration. The non-linear PT solver should look like:

```julia
ϵ    = 1e-8
er   = Inf
iter = 0
while er > ϵ
    iter == itermax && break
    tot_iter += 1
    iter     += 1
    h_w      .= ??
    K        .= ??
    Re        = @. π + √(π^2 + L^2 / K / dt)
    Δτ_θ      = @. Vpdτ * L / K / Re
    α_Δτ      = @. L / Vpdτ / Re
    qx       .= ??
    qy       .= ??
    R        .= ??
    @views h[2:end-1, 2:end-1] .= ??
    # check residual
    er        = norm(R) / n^2
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

