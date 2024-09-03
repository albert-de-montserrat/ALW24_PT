```@meta
EditURL = "../scripts/7_Helperfunctions.jl"
```

# Helperfunctions
Functions that can be used during the coding sessions of the course.
They can be make the functions you write more readable and concise.

```julia
inn(A)   = @views A[2:end-1, 2:end-1]
inn_x(A) = @views A[2:end-1, :]
inn_y(A) = @views A[:, 2:end-1]
av_x(A)  = @views @. 0.5 * (A[1:end-1, :] + A[2:end, :])
av_y(A)  = @views @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])
av_xi(A) = @views av_x(A)[:, 2:end-1]
av_yi(A) = @views av_y(A)[2:end-1, :]
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

