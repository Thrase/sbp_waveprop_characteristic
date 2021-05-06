using SparseArrays
using LinearAlgebra
using SparseArrays

include("sim_funs.jl")
include("square_circle.jl")

include("mms.jl")

let
    # Define the friction law to use
    friction(V) = 2asinh(V)

    sbp_order = 4

    # This is the base mesh size in each dimension on each element.
    N0 = 17

    refinement_levels = 3

    # The element to domain map is needed here in order to define the boundary
    # functions (we have a circular argument going on that we need EToDomain to
    # evaluate the boundary conditions, but we also need the boundary conditions
    # to build the square_circle operators)
    bc_map =
        [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN, BC_JUMP_INTERFACE]

    (_, _, _, _, EToDomain) = read_inp_2d("square_circle.inp"; bc_map = bc_map)

    # Boundary condition function defined from the mms solution
    gDfun(x, y, t, e) = ue(x, y, t, EToDomain[e])
    gDdotfun(x, y, t, e) = ue_t(x, y, t, EToDomain[e])
    function gNfun(nx, ny, xf, yf, t, e)
        dom = EToDomain[e]
        return nx .* (ue_x(xf, yf, t, dom)) + ny .* (ue_y(xf, yf, t, dom))
    end

    body_force(x, y, t, e) = force(x, y, t, EToDomain[e])

    # Loop over the levels of refinement
    ϵ = zeros(refinement_levels)
    for lvl in 1:length(ϵ)
        # Set up the local grid dimensions
        Nr = N0 * (2^(lvl - 1))
        Ns = N0 * (2^(lvl - 1))

        Nqr = Nr + 1
        Nqs = Ns + 1

        Np = Nqr * Nqs

        # Create the operators for the problem
        (metrics, rhsops, EToDomain, EToF, EToO, EToS, FToB, FToE, FToLF) =
            build_square_circle(
                sbp_order,
                Nqr,
                Nqs,
                1,
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
                bc_map,
                lvl == 1,
            )
        nelems = length(rhsops)

        # initial conditions
        lenq0 = 2 * Np + 2 * Nqr + 2 * Nqs
        q = Array{Float64, 1}(undef, nelems * lenq0)
        for e in 1:nelems
            (xf1, xf2, xf3, xf4) = metrics[e].facecoord[1]
            (yf1, yf2, yf3, yf4) = metrics[e].facecoord[2]

            dom = EToDomain[e]
            u0 = ue.(metrics[e].coord[1][:], metrics[e].coord[2][:], 0, dom)
            v0 = ue_t.(metrics[e].coord[1][:], metrics[e].coord[2][:], 0, dom)
            û10 = ue.(xf1[:], yf1[:], 0, dom)
            û20 = ue.(xf2[:], yf2[:], 0, dom)
            û30 = ue.(xf3[:], yf3[:], 0, dom)
            û40 = ue.(xf4[:], yf4[:], 0, dom)

            q[(lenq0 * (e - 1) + 1):(e * lenq0)] =
                [u0; v0; û10; û20; û30; û40]
        end

        # solve the ODE
        tspan = (0.0, 1.0)
        hmin = mapreduce(m -> m.hmin, min, values(metrics))
        dt = 2hmin
        params = (
            Nqr = Nqr,
            Nqs = Nqs,
            rhsops = rhsops,
            EToF = EToF,
            EToO = EToO,
            EToS = EToS,
            FToB = FToB,
            FToE = FToE,
            FToLF = FToLF,
            friction = friction,
        )
        timestep!(q, waveprop!, params, dt, tspan)

        # Loop over the blocks and compute the error in each block
        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:Np]

            dom = EToDomain[e]
            qexact_u =
                ue.(
                    metrics[e].coord[1][:],
                    metrics[e].coord[2][:],
                    tspan[2],
                    dom,
                )
            qexact_v =
                ue_t.(
                    metrics[e].coord[1][:],
                    metrics[e].coord[2][:],
                    tspan[2],
                    dom,
                )
            qexact_û1 =
                ue.(
                    metrics[e].facecoord[1][1],
                    metrics[e].facecoord[2][1],
                    tspan[2],
                    dom,
                )
            qexact_û2 =
                ue.(
                    metrics[e].facecoord[1][2],
                    metrics[e].facecoord[2][2],
                    tspan[2],
                    dom,
                )
            qexact_û3 =
                ue.(
                    metrics[e].facecoord[1][3],
                    metrics[e].facecoord[2][3],
                    tspan[2],
                    dom,
                )
            qexact_û4 =
                ue.(
                    metrics[e].facecoord[1][4],
                    metrics[e].facecoord[2][4],
                    tspan[2],
                    dom,
                )

            Δu = qe_u - qexact_u
            Mv = rhsops[e].JH

            ϵ[lvl] += Δu' * Mv * Δu
        end # end compute error at lvl

        ϵ[lvl] = sqrt(ϵ[lvl])
        @show (lvl, ϵ[lvl])
    end #loop over levels

    println((log.(ϵ[1:(end - 1)]) - log.(ϵ[2:end])) / log(2))
end
