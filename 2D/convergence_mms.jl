include("run_mms.jl")

let
    N0 = 17
    for sbp_order in (2, 4, 6)
        @show sbp_order
        main(
            sbp_order,
            4,
            N0;
            characteristic_method = true,
            cfl = 2,
            friction = (V) -> 128 * asinh(V),
            tspan = (0.0, 1.0),
            do_output = true,
        )
    end
end
