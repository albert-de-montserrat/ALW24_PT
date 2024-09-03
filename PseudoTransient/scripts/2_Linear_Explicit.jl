# # Hillslope diffusion
# Subaerial slope diffusion can be modeled with the diffusion equation.

# $$
# \frac{\partial h}{\partial t} = \frac{\partial}{\partial x} \left(-q \right)
# $$

# $$
# q = -K \frac{\partial h}{\partial x}
# $$

# where $q$ is the flux and $K$ is the subaerial hillslope diffusion. For now, we will consider $K$ to be a constant value for the whole domain.

# ## Temporal discretization
# ### Implicit

# $$
# \frac{h^{t+1} - h^t}{\Delta t} = \frac{\partial}{\partial x} \left(K \frac{\partial h^{t+1}}{\partial x} \right)
# $$

# Since $h^{t+1}$ depends on its spatial derivative, rearrangement of the previous equation leads to a system of linear equations $A x = b$, where $A$ is a sparse-matrix, and $x$ and $b$ are column vectors. This temporal discretization is usually rather stable, however building the sparse-matrix $A$ and solving the linear system of equations is usually computationally rather expensive.

# ### Explicit

# $$
# \frac{h^{t+1} - h^t}{\Delta t} = \frac{\partial}{\partial x} \left(K \frac{\partial h^t}{\partial x} \right)
# $$

# which leads to

# $$
# h^{t+1} = h^t + \Delta t  \frac{\partial}{\partial x} \left(K \frac{\partial h^t}{\partial x} \right)
# $$

# Solving the previous equation straightforward as it is matrix-free. This means that it does not require to build a sparse matrix for the left-hand-side and we do not have to do a linear solve. This is nice because it is easy to implement and computationally efficient; however, this introduces limiations on the maximum time step allowed, whose upper bound if given by the CFL condition

# $$
# \frac{K \Delta t}{\Delta x}<C_{max}
# $$

# where

# $$
# C_{max}\leq 1
# $$

# ## Spatial discretization

# The spatial derivatives can be discretization with different methods: Finite Differences, Finite Elements, Finite Volumes, etc. In this workshop we will discretize the equations with Finite Differences with a staggered grid.

# <p align="center">
# <img src="../figs/2Dgrid.png" alt="drawing" width="300"/>
# </p>

# ### Explicit

# $$
# q_{i+\frac{1}{2}} = -K \frac{h_{i+1}^t - h_{i}^t}{\Delta x}
# $$

# $$
# h^{t+1}_i = h^t_i - \Delta t \frac{q_{i+\frac{1}{2}} - q_{i-\frac{1}{2}}}{\Delta x}
# $$

# # 💾 Time to program 💾
# We will first implement a simple explicit solver for hillslope diffusion
# We will start by defining the geometry:

n       = 128
L       = 50e3  # length of the topography profile
x       = LinRange(0, L, n)
y       = LinRange(0, L, n)
dx      = x[2] - x[1]
dy      = y[2] - y[1]
X       = [x for x in x, y in y]
Y       = [y for x in x, y in y]
h       = zeros(n, n)
ind     = @. (20e3 ≤ X ≤ 30e3) && (20e3 ≤ Y ≤ 30e3)
h[ind] .= 500

# Note that the topography is rather arbitrary and we could use any other profile. Anyhow, we can visualise it with `GLMakie.jl` (or `CairoMakie.jl`)

using CairoMakie
fig, ax, m = surface(x ./ 1e3, y ./ 1e3, h./1e2,colormap=:lipari)
Colorbar(fig[1,2], m)
fig

# 🔔 **Optional** : load a topography file using GeophysicalModelGenerator.jl and GMT

using GeophysicalModelGenerator, GMT
using Interpolations, LinearAlgebra

# Since we are in France, let's import the topography of the Montblanc from the GMT's server and extract 2D array from the data structure

topography_GMT = import_topo([6.35, 7.35, 45.35, 46.35], file="@earth_relief_03s") 
surf = topography_GMT.depth.val[:,:,1]

# Now we need to convert the coordinate system from lat-long to meters in a Cartesian box

x = LinRange(6.351, 7.349, n)
y = LinRange(45.351, 46.349, n)
X = topography_GMT.lon.val[:, 1, 1], topography_GMT.lat.val[1, :, 1]
itp = interpolate(X, surf, Gridded(Linear()))
h   = [itp(x, y) * 1e3 for x in x, y in y]
heatmap(x, y, h, colormap=:oleron, colorrange=(-4810, 4810))

# compute the x and y size of our cartesian model
lat_dist  = extrema(X[1]) |> collect |> diff |> first |> abs
long_dist = extrema(X[2]) |> collect |> diff |> first |> abs
Lx        = 1e3 * lat_dist * 110.574
Ly        = 1e3 * long_dist * 111.320 * cos(lat_dist)

# and finally define the geometry again
x       = LinRange(0, Lx, n)
y       = LinRange(0, Ly, n)
dx      = x[2] - x[1]
dy      = y[2] - y[1]
X       = [x for x in x, y in y]
Y       = [y for x in x, y in y]

# We only have to define a single constant physical paremeter, the hillslope diffusion coefficient

K   = 2.5 # [m^2/year]

# Remember that the time stepping is now limiter by the CFL condition

C  = 0.1
dt = C * dx^2 / K # [yrs]

# Taking a look at the equations, we only need three two-dimensional `Array`s to solve the explicit problem:

# |     Array      |        size      |
# |:--------------:|:----------------:|
# | $\mathbf{h}$   | $n   \times n  $ |
# | $\mathbf{q}_x$ | $n-1 \times n-2$ |
# | $\mathbf{q}_y$ | $n-2 \times n-1$ |

# where $n$ is the number of grid points. We have already created $\mathbf{h}$ as the initial topography profile `h`. We just need to allocate the fluxes $\mathbf{q}_x$ and $\mathbf{q}_y$

qx              = zeros(n-1, n-2)
qy              = zeros(n-2, n-1)
initial_profile = copy(h) # save initial profile for plotting purposes

# We now have all the objects needed to solve the hillslope diffusion equation, we just need to decide for how long we want to run the model and setup the time stepping

# ```julia
# run_time = 1e6 # [yr]
# t        = 0
# nt       = 0
# while t < run_time
#     qx .= ??
#     qy .= ??
#     @views h[2:end-1, 2:end-1] .= ??
#     nt += 1
#     t  += dt
# end
# ```

# ## Exercise
# 1. Define the missing functions `compute_flux` and `update_h` to compute the fluxes `q` and update `h`, respectively.
# 2. How much can we increase `C` before the model 🧨💥?