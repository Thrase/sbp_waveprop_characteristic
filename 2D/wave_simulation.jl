using SparseArrays
using LinearAlgebra
using SparseArrays
using Printf

include("sim_funs.jl")
include("square_circle.jl")

let
    friction(V) = 2asinh(V)

    sbp_order = 6

    # This is the base mesh size in each dimension on each element.
    N0 = 48

    # solver times (output done at each break point)
    ts = 0:0.1:10

    bc_map =
        [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN, BC_JUMP_INTERFACE]

    # Set dirichlet data at left/right boundaries
    μ1 = 0.1
    μ2 = 0.2
    σ1 = 0.005
    σ2 = 0.005
    ue(x, y, _) = exp.(-(x .- μ1) .^ 2 ./ σ1 .- (y .- μ2) .^ 2 ./ 2σ2)
    ue_t(_...) = 0
    gDfun(_...) = 0
    gDdotfun(_...) = 0
    gNfun(_...) = 0
    body_force = nothing

    Nr = Ns = N0

    Nqr = Nr + 1
    Nqs = Ns + 1

    Np = Nqr * Nqs

    λ1(x, y) = (1 + sin(x)^2) / 2
    λ2(x, y) = exp(-(x + y)^2)
    θ(x, y) = 2 * π * sin(x) * sin(y)
    cxx(x, y) = cos(θ(x, y))^2 * λ1(x, y) + sin(θ(x, y))^2 * λ2(x,y)
    cxy(x, y) = cos(θ(x, y)) * sin(θ(x, y)) * (λ2(x,y) - λ1(x,y))
    cyy(x, y) = sin(θ(x, y))^2 * λ1(x, y) + cos(θ(x, y))^2 * λ2(x, y)

    (metrics, rhsops, EToDomain, EToF, EToO, EToS, FToB, FToE, FToLF) =
        build_square_circle(
            sbp_order,
            Nqr,
            Nqs,
            gDfun,
            gDdotfun,
            gNfun,
            body_force,
            bc_map,
            true;
            cxx = cxx,
            cxy = cxy,
            cyy = cyy
        )

    nelems = length(rhsops)

    # initial conditions
    lenq0 = 2 * Np + 2 * Nqr + 2 * Nqs
    q = Array{Float64, 1}(undef, nelems * lenq0)
    for e in 1:nelems
        (xf1, xf2, xf3, xf4) = metrics[e].facecoord[1]
        (yf1, yf2, yf3, yf4) = metrics[e].facecoord[2]

        u0 = ue.(metrics[e].coord[1][:], metrics[e].coord[2][:], e)
        v0 = ue_t.(metrics[e].coord[1][:], metrics[e].coord[2][:], e)
        û10 = ue.(xf1[:], yf1[:], e)
        û20 = ue.(xf2[:], yf2[:], e)
        û30 = ue.(xf3[:], yf3[:], e)
        û40 = ue.(xf4[:], yf4[:], e)

        q[(lenq0 * (e - 1) + 1):(e * lenq0)] = [u0; v0; û10; û20; û30; û40]
    end

    mkpath("output")
    write_vtk(@sprintf("output/N_blocks_sim_step_%04d", 0), metrics, q;
              cxx = cxx, cxy = cxy, cyy = cyy)

    hmin = mapreduce(m -> m.hmin, min, values(metrics))
    dt = 2hmin

    energy = zeros(length(ts))

    # Parameters to pass to the ODE solver
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

    # Loop over times and advance simulation (saving at given times before
    # continuing)
    for step in 1:(length(ts) - 1)
        tspan = (ts[step], ts[step + 1])
        @show tspan
        timestep!(q, waveprop!, params, dt, tspan)
        write_vtk(@sprintf("output/N_blocks_sim_step_%04d", step), metrics, q;
              cxx = cxx, cxy = cxy, cyy = cyy)

        # Loop over blocks are compute the energy in the blocks at this time
        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:Np]
            qe_v = qe[Np .+ (1:Np)]
            qe_û1 = qe[(2Np + 1):(2Np + Nqs)]
            qe_û2 = qe[(2Np + 1 + Nqs):(2(Np + Nqs))]
            qe_û3 = qe[(2(Np + Nqs) + 1):(2(Np + Nqs) + Nqr)]
            qe_û4 = qe[(2(Np + Nqs) + Nqr + 1):(2(Np + Nqs + Nqr))]

            τ̂ = (
                rhsops[e].nCB[1] * qe_u + 1 * rhsops[e].nCnΓ[1] * qe_û1 -
                1 * rhsops[e].nCnΓL[1] * qe_u,
                rhsops[e].nCB[2] * qe_u + 1 * rhsops[e].nCnΓ[2] * qe_û2 -
                1 * rhsops[e].nCnΓL[2] * qe_u,
                rhsops[e].nCB[3] * qe_u + 1 * rhsops[e].nCnΓ[3] * qe_û3 -
                1 * rhsops[e].nCnΓL[3] * qe_u,
                rhsops[e].nCB[4] * qe_u + 1 * rhsops[e].nCnΓ[4] * qe_û4 -
                1 * rhsops[e].nCnΓL[4] * qe_u,
            )

            Mu = rhsops[e].Ã
            Mv = rhsops[e].JH

            Mû1 = rhsops[e].H[1]
            Mû2 = rhsops[e].H[2]
            Mû3 = rhsops[e].H[3]
            Mû4 = rhsops[e].H[4]

            Mτ̂1 = rhsops[e].X[1] * rhsops[e].H[1]
            Mτ̂2 = rhsops[e].X[2] * rhsops[e].H[2]
            Mτ̂3 = rhsops[e].X[3] * rhsops[e].H[3]
            Mτ̂4 = rhsops[e].X[4] * rhsops[e].H[4]

            Mu1 =
                (rhsops[e].nCB[1])' *
                rhsops[e].X[1] *
                rhsops[e].H[1] *
                rhsops[e].nCB[1]
            Mu2 =
                (rhsops[e].nCB[2])' *
                rhsops[e].X[2] *
                rhsops[e].H[2] *
                rhsops[e].nCB[2]
            Mu3 =
                (rhsops[e].nCB[3])' *
                rhsops[e].X[3] *
                rhsops[e].H[3] *
                rhsops[e].nCB[3]
            Mu4 =
                (rhsops[e].nCB[4])' *
                rhsops[e].X[4] *
                rhsops[e].H[4] *
                rhsops[e].nCB[4]

            energy[step] +=
                0.5 * qe_v' * Mv * qe_v +
                0.5 * qe_u' * Mu * qe_u +
                0.5 * (
                    1 * τ̂[1]' * Mτ̂1 * τ̂[1] - 1 * qe_u' * Mu1 * qe_u +
                    1 * τ̂[2]' * Mτ̂2 * τ̂[2] - 1 * qe_u' * Mu2 * qe_u +
                    1 * τ̂[3]' * Mτ̂3 * τ̂[3] - 1 * qe_u' * Mu3 * qe_u +
                    1 * τ̂[4]' * Mτ̂4 * τ̂[4] - 1 * qe_u' * Mu4 * qe_u
                )
        end
    end
end
