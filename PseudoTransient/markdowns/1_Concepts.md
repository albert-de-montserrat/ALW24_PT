```@meta
EditURL = "../scripts/1_Concepts.jl"
```

# Instantiating arrays
Julia offers native support for multi-dimensional arrays and "array programming" similar to MATLAB.
There are different ways to instantiate arrays; for example, you can initialize an empty $4 \times 4$ array that will
contain only double precision floats (`Float64`) as

````@example 1_Concepts
A = Array{Float64, 2}(undef, 4, 4)
````

In this particular case, `Matrix{T}` is an alias type of `Array{Float64, 2}`, so you can also initialize it as

````@example 1_Concepts
A = Matrix{Float64}(undef, 4, 4)
````

Other common `Array` constructors are `zeros`, `ones`, `fill`, and `rand`

````@example 1_Concepts
A = zeros(4, 4)
````

````@example 1_Concepts
A = ones(4, 4)
````

````@example 1_Concepts
A = fill(4, 4)
````

````@example 1_Concepts
A = rand(4, 4)
````

# Array indexing and mutation
To read individual elements of an array, we use square brackets

````@example 1_Concepts
A = rand(4)
A[1]
````

````@example 1_Concepts
B = rand(4, 4)
B[1, 1]
````

````@example 1_Concepts
C = rand(4, 4, 4)
C[1, 1, 1]
````

We can also read chunks of the array at once (_NOTE_ this will allocate a new array)

````@example 1_Concepts
B[1:2, 1:2]
````

To make a view of an array and avoid allocating new intermediate arrays, we can use the macro `@views`

````@example 1_Concepts
@views B[1:2, 1:2]
````

As in MATLAB, we can also slice `Array`s

````@example 1_Concepts
@views B[:, 1] # take the first column
````

````@example 1_Concepts
@views B[1, :] # take the first row
````

# Broadcasting
`Array`s support element-wise (broadcasting) linear algebra operators, we only need to add a dot `.` before the operator. For example

````@example 1_Concepts
A = rand(4, 4)
B = rand(4, 4)
C = A .+ B
````

````@example 1_Concepts
D = A ./ B
````

If we need to broadcast several operators in a single line, we can fuse them with the `@.` macro

````@example 1_Concepts
E = @. (A + B) * B^2
````

Broadcasting does not apply only to operators, but to any function that acts on scalars. For example, we can broadcast `sin` to any `Array`

````@example 1_Concepts
sin.(A)
````

# Functions
Functions can be instantiated in several ways, with the "standard" way being

````@example 1_Concepts
function foo(x)
    y = x + rand()
    return y
end
````

A function (or code block, i.e. `if/elseif` statements or `let` blocks) will return their **last line**. Therefore, we can also write the previous function as

````@example 1_Concepts
function foo(x)
    y = x + rand()
    y
end
````

````@example 1_Concepts
function foo(x)
    y = x + rand()
end
````

or more concisely as a one-liner

````@example 1_Concepts
foo(x) = x + rand()
````

Functions can also return multiple arguments

````@example 1_Concepts
function foo(x)
    y = x + rand()
    return x, y
end
````

Anonymous functions also exist in Julia

````@example 1_Concepts
fun = x -> x + rand()
fun(5)
````

All the information regarding functions can be found [here](https://docs.julialang.org/en/v1/manual/functions/)

## In-place vs out-of-place functions
It is common that you need to store the results of matrix operations in a new array. Creating the new destination array will obviously allocate, if you need to do this operation just once, an out-of-place kernel will do just fine

````@example 1_Concepts
function outofplace(A, B)
    C = similar(A)
    for i in axes(A, 1), j in axes(A, 2)
        C[i, j] = B[i, j] * C[i, j]
    end
    return C
end
````

however, it is frequent you need to run the kernel several times, in which case you want to avoid allocating the destination array every time. In this case, you can use an in-place kernel, by pre-allocating the destination array and just mutating it:

````@example 1_Concepts
function inplace!(C, A, B)
    for i in axes(A, 1), j in axes(A, 2)
        C[i, j] = B[i, j] * C[i, j]
    end
    return nothing
end
````

note that in the latter case we do not return anything, as Julia passes arguments by reference to the functions.
Also note the `!` at the end for the in-place function, this is a convention in Julia to denote that the function mutates its arguments (where the mutating arguments are the first ones).

# Benchmarking
In Julia, it is extremely easy (and extremely useful to catch performance issues or regressions) to benchmark the performance of any function.
Julia has the built-in `@time` and `@elapsed` macros that measure the performance of a single function call. However, these are not
very accurate and it's best to use external packages, namely [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/) or [Chairmarks](https://github.com/LilithHafner/Chairmarks.jl)
We can benchmark our `outofplace` and `inplace!` functions with `BenchmarkTools.jl`

````@example 1_Concepts
using BenchmarkTools
A, B, C = rand(32, 32), rand(32, 32), zeros(32, 32)
@benchmark outofplace($A, $B)
````

````@example 1_Concepts
@benchmark inplace!($C, $A, $B)
````

or with `Chairmarks.jl`

````@example 1_Concepts
using Chairmarks
@be outofplace($A, $B)
````

````@example 1_Concepts
@be inplace!($C, $A, $B)
````

# Code parallelization
Julia supports parallel computing on both CPUs and GPUs throught different packages
- CPU:
  1. [Threads.jl](https://docs.julialang.org/en/v1/manual/multi-threading/) for multithreading (i.e. shared memory)
  2. [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) for distributed parallelization (i.e. across multiple CPUs)
  3. [MPI.jl](https://github.com/JuliaParallel/MPI.jl) for distributed parallelization (i.e. across multiple CPUs)
- GPU:
  1. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for Nvidia GPU cards
  2. [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) for AMD GPU cards
  3. [Metal.jl](https://github.com/JuliaGPU/Metal.jl) for Apple M-series chips

## Multithreading
Julia does not parallelize code automatically, but it provides tools to do so. This means that "array-style programming" ala MATLAB or Numpy will not automatically parallelize in Julia.
However, is is very simple to paralleliza loops (and **loops are fast**) in Julia.
For example, let's parallelize the AXPY ("A·X Plus Y") BLAS kernel

$$A = \alpha B + C$$

where $B$ and $C$ are random $n\times n$ matrices.

````@example 1_Concepts
n    = 128
α    = rand()
Z    = zeros(n, n)
X, Y = rand(n, n), rand(n, n)
````

To parallelize this operation, we simply need to write the kernel as a loop and the put the `Threads.@threads` macro before the loop

````@example 1_Concepts
function axpy!(Z, X, Y, α)
    @assert size(Z) == size(X) == size(Y)
    Threads.@threads for i in eachindex(X)
        Z[i] = α * X[i] + Y[i]
    end
end
````

## GPU programming
Unlike with CPU code, broadcasting works and one can run GPU-accelerated  "array-style programming" code, as long as the arrays are properly allocated on the GPU device. For example, the AXPY operation can be run on the GPU of a M-series chip as follows:

````@example 1_Concepts
using Metal
n    = 128
α    = rand(Float32)
Z    = Metal.zeros(n, n)
X, Y = Metal.rand(n, n), Metal.rand(n, n)
Z    = @. α * X + Y
````

If you have a NVIDIA or AMD GPU card, you can respectively use the CUDA.jl or AMDGPU.jl packages to run the same code just using the `CUDA.zeros`, `CUDA.rand` or `AMDGPU.zeros`, `AMDGPU.rand` allocators instead.
However, it is often more convinient to write function kernels, as not everything can be efficiently written an array programming style.

<p align="center">
<img src="../figs/cuda_grid.png" alt="drawing" width="300"/>
</p>

Using first linear indexing, the previous example would be written as follows

````@example 1_Concepts
function axpy!(Z, X, Y, α)
    i = thread_position_in_grid_1d()
    if i ≤ length(Z)
        Z[i] = α * X[i] + Y[i]
    end
    return
end

nthreads = 512
ngroups  = cld(n^2, nthreads)
@metal threads = nthreads groups=ngroups axpy!(Z, X, Y, α)
````

Or like this if we prefer to use Cartesian indexing:

````@example 1_Concepts
function axpy_Metal!(Z, X, Y, α)
    (i,j) = thread_position_in_grid_2d()
    if i ≤ size(Z, 1) && j ≤ size(Z, 2)
        Z[i, j] = α * X[i, j] + Y[i, j]
    end
    return
end

nthreads = 16, 16
ngroups  = cld.(n, nthreads)
Metal.@sync @metal threads = nthreads groups=ngroups axpy_Metal!(Z, X, Y, α)
````

> [!WARNING]
> Multithreaded and GPU code **is not thread safe**. It is up to the user to avoid race conditions. An example of non thread-safe code is a reduction:

````@example 1_Concepts
let
    A = rand(20)
    sum = 0.0
    Threads.@threads for Ai in A
        sum += Ai # <- race condition
    end
end
````

# Backend abstraction
As you can see, if we want to write a code that is highly portable and run it multiple backends, we need to write a specific kernel for each hardware.
This is tedious and leads to a lot of code redudancy and prone to copy-pasting bugs. One of the advantages of Julia, is that we can easily write code that is agnostic to the backend.
This is done using one of these packages:
  - [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl); supports CUDA.jl, AMDGPU.jl, and Metal.jl.
  - [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl); supports CUDA.jl and AMDGPU.jl.

## ParallelStencil.jl
Kernel parallelization is essentially done using two new macros: `@parallel` and `@parallel_indices`

````@example 1_Concepts
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) # @init_parallel_stencil(Backend, TypePrecission, Dimension)

n    = 128
α    = rand()
Z    = @zeros(n, n)
X, Y = @rand(n, n), @rand(n, n)

@parallel_indices (i,j) function axpy_PS!(Z, X, Y, α)
    Z[i, j] = α * X[i, j] + Y[i, j]
    return
end

@parallel (1:n, 1:n) axpy_PS!(Z, X, Y, α)
````

## KernelAbstractions.jl
This package requires a bit more written, as now we need to define a kernel and a launcher function to parallelize our code

````@example 1_Concepts
using Metal # comment out to run on the CPU; or load CUDA / AMDGPU instead
using KernelAbstractions, Random

backend = MetalBackend() # other valid backends: CPU(), CUDABackend(), AMDGPUBackend()
type    = Float32
n       = 128
α       = rand(type)
Z       = KernelAbstractions.zeros(backend, type, n, n)
X       = rand!(allocate(backend, type, n, n))
Y       = rand!(allocate(backend, type, n, n))
````

kernel

````@example 1_Concepts
@kernel function axpy_KA_kernel!(Z, X, Y, α)
    i, j = @index(Global, NTuple)
    Z[i, j] = α * X[i, j] + Y[i, j]
end
````

launcher

````@example 1_Concepts
function axpy_KA!(Z, X, Y, α)
    backend = get_backend(Z)
    @assert size(X) == size(Y) == size(Z)
    @assert get_backend(X) == get_backend(Y) == backend

    nthreads = 32, 32
    kernel = axpy_KA_kernel!(backend, nthreads)
    kernel(Z, X, Y, α, ndrange = size(Z))
    KernelAbstractions.synchronize(backend)
    return
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

