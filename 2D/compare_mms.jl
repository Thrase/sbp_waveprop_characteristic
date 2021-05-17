include("run_mms.jl")

let
    N0 = 48
    αs = (1, 4, 16, 64, 128)
    @show cfls = 2.0 .^ (0:-1:-5)

    for sbp_order in (2, 4, 6)
        for characteristic_method in (true, false)
            for α in αs
                @show (sbp_order, characteristic_method, α)
                for cfl in cfls
                    ϵ = main(
                        sbp_order,
                        1,
                        N0;
                        characteristic_method = characteristic_method,
                        cfl = cfl,
                        friction = (V) -> α * asinh(V),
                        tspan = (0.0, 0.1),
                        do_output = false,
                    )[1]
                    @show (cfl, ϵ)
                end
                println()
            end
        end
    end
end
