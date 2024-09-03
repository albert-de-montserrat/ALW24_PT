# 2024 Ada Lovelace Workshops on Modelling Mantle and Lithosphere Dynamics
[![](figs/AdaLovelace.png)](https://meetings.copernicus.org/2024AdaLovelaceWorkshop/)

## Program

- Brief **intro to Julia**
  - Installation and running interactive sessions with VSCode
  - Basic Julia programming:
    - Arrays, functions,...
- **Hands-on** - solving forward problems with the Pseudo-Transient method
  - Linear and non-linear diffusion problem
  - From a explicit solver to the (accelerated) Pseudo-Transient method
  - Parallelizing the CPU code and porting it to GPUs

## Overview

The main Julia packages we will rely on are:
- Running Jupyter notebooks with the Julia kernel
  - [IJulia.jl]("https://github.com/JuliaLang/IJulia.jl")
- GPU packages, only one (or none if you dont have access to a GPU) of the following is needed:
  - [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for those with an Nvidia GPU
  - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) for those with an AMD GPU
  - [Metal.jl](https://github.com/JuliaGPU/Metal.jl) for those with a Macbook with a M-series chip
- Parallel programming
  - [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
  - [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- Visualization  
  - [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) or [GLMakie.jl](https://github.com/MakieOrg/Makie.jl) for plotting
- Optional
  - [GeophysicalModelGenerator.jl](https://github.com/JuliaGeodynamics/GeophysicalModelGenerator.jl)
  - [GMT.jl]("https://github.com/GenericMappingTools/GMT.jl")
  - [Interpolations.jl]("https://github.com/JuliaMath/Interpolations.jl")
- If there is time...
  - [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl)
  - [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl)
  - [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)

If you want to install all these packages (except CUDA.jl and AMDGPU.jl), open a Julia session and type in the REPL
```julia-repl
import Pkg; Pkg.activate("."); Pkg.instantiate()
```

#### :bulb: Useful extra resources
- The Julia language docs: [https://docs.julialang.org/en/v1/](https://docs.julialang.org/en/v1/)
- Julia Discourse (Julia Q&A): [https://discourse.julialang.org](https://discourse.julialang.org)
- Julia Slack (Julia dev chat): [https://julialang.org/slack/](https://julialang.org/slack/)
- PDE on GPUs ETH Zurich course: [https://pde-on-gpu.vaw.ethz.ch](https://pde-on-gpu.vaw.ethz.ch)

# Installing Julia

## Download binary

Download the Julia binary from the official [website](https://julialang.org/downloads/). The binary is available for Windows, macOS, and Linux.

- **Recommended version** : _Current stable release: v1.10.4 (June 4, 2024)_
- Install a **64-bits** binary
- Do **not** install v1.11, several packages will not work

##  installation via CLI

- Windows: 

>`winget install julia -s msstore`

- Mac & Linux:

> `curl -fsSL https://install.julialang.org | sh`

##  Juliaup - version manager (**RECOMMENDED**)

Juliaup is a version manager that allows you to easily install and switch to different Julia versions.

### Juliaup installation

- Windows: 

> `winget install julia -s msstore`

- Mac & Linux: 

> `curl -fsSL https://install.julialang.org | sh`

### Julia installation

You can install any version of Julia with the Juliaup CLI tools. For example, to install julia-1.10.4 open your terminal and type

>`juliaup add 1.10.4`

If you have several Julia versions installed, you can switch versions by openning julia as e.g.

>`julia +1.9.1`

or make it the default version as

>`juliaup default 1.9.1`

Check `juliaup` [GitHub page](https://github.com/JuliaLang/juliaup) for more config options.

# Using Julia

The recommended way of developing code in Julia is using the [VSCode](https://code.visualstudio.com/) IDE.

## Julia with VSCode

### Installing Julia plug-in in VSCode
![](Part1/figs/VSC_plugin.png)

### Starting an interactive Julia session in VSCode
![](Part1/figs/InteractiveSession1.png)

![](Part1/figs/InteractiveSession2.png)

## Julia from the terminal

### Interactive session

1. Add `julia` binary to your environment path
2. Open terminal
3. Type `julia` to open the REPL

### Launching a script
1. Add `julia` binary to your environment path
2. Open terminal
3. Type `julia myScript.jl`. More info [here](https://docs.julialang.org/en/v1/manual/command-line-interface/).


# Installing registered packages

Packages can easily be installed with Julia's package manager. To enter the package manager mode, simply open an interactive julia session and type `]` in the REPL. Once inside the package manager a package can be installed by typing `add MyPkg`. For example, to install `GLMakie.jl`, one of the plotting packages, just type
```julia-repl
] add GLMakie
```

Specific versions can also be insalled, for example, to install v0.5 of GLMakie
```julia-repl
] add GLMakie@0.5
```
and also specific branches from the host repository
```julia-repl
] add GLMakie#myBranch
```

Packages can also be installed outside the package manager in a more verbose way
```julia-repl
import Pkg; Pkg.add("GLMakie")
```
