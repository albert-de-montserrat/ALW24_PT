using Literate

fns  = Literate.markdown, Literate.notebook
outs = "PseudoTransient/markdowns", "PseudoTransient/notebooks" 
ins  = readdir("PseudoTransient/scripts/", join=true)

for file in ins
    for (fn, out) in zip(fns, outs)
        fn(
            f,
            out
        )
    end
end

# fns  = (Literate.notebook, )
# outs = ("Part1/notebooks", )

# for (fn, out) in zip(fns, outs)
#     fn(
#         "Part1/scripts/6_Parallelization.jl",
#         out
#     )
# end