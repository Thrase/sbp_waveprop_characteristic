using SparseArrays
using LinearAlgebra
using SparseArrays
using Printf
using DelimitedFiles

include("sim_funs.jl")
include("square_circle.jl")

function compute_energy(q, rhsops, Np)
    nelems = length(rhsops)
    energy = 0
    Nq = round(Int, sqrt(Np))
    @assert Np == Nq^2
    lenq0 = 2 * Np + 2 * Nq + 2 * Nq

    # Loop over blocks are compute the energy in the blocks at this time
    for e in 1:nelems
        qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

        qe_u = qe[1:Np]
        qe_v = qe[Np .+ (1:Np)]
        qe_û1 = qe[(2Np + 1):(2Np + Nq)]
        qe_û2 = qe[(2Np + 1 + Nq):(2(Np + Nq))]
        qe_û3 = qe[(2(Np + Nq) + 1):(2(Np + Nq) + Nq)]
        qe_û4 = qe[(2(Np + Nq) + Nq + 1):(2(Np + Nq + Nq))]

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

        energy +=
            0.5 * qe_v' * Mv * qe_v +
            0.5 * qe_u' * Mu * qe_u +
            0.5 * (
                1 * τ̂[1]' * Mτ̂1 * τ̂[1] - 1 * qe_u' * Mu1 * qe_u +
                1 * τ̂[2]' * Mτ̂2 * τ̂[2] - 1 * qe_u' * Mu2 * qe_u +
                1 * τ̂[3]' * Mτ̂3 * τ̂[3] - 1 * qe_u' * Mu3 * qe_u +
                1 * τ̂[4]' * Mτ̂4 * τ̂[4] - 1 * qe_u' * Mu4 * qe_u
            )
    end
    return sqrt(energy)
end

let
    friction(V) = asinh(V)

    sbp_order = 6
    cfl = 1 / 2

    data = Dict()

    # This is the base mesh size in each dimension on each element.
    Nvals = 17 * 2 .^ (3:-1:1)
    for N in Nvals

        # solver times (output done at each break point)
        ts = 0:0.01:3

        bc_map = [
            BC_DIRICHLET,
            BC_DIRICHLET,
            BC_NEUMANN,
            BC_NEUMANN,
            BC_JUMP_INTERFACE,
        ]

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

        Nq = N + 1

        Np = Nq * Nq

        λ1(x, y) = 1
        λ2(x, y) = 1 / 2
        θ(x, y) = π * (2 - x) * (2 - y) / 4
        cxx(x, y) = cos(θ(x, y))^2 * λ1(x, y) + sin(θ(x, y))^2 * λ2(x, y)
        cxy(x, y) = cos(θ(x, y)) * sin(θ(x, y)) * (λ2(x, y) - λ1(x, y))
        cyy(x, y) = sin(θ(x, y))^2 * λ1(x, y) + cos(θ(x, y))^2 * λ2(x, y)

        (metrics, rhsops, EToDomain, EToF, EToO, EToS, FToB, FToE, FToLF) =
            build_square_circle(
                sbp_order,
                Nq,
                Nq,
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
                bc_map,
                true;
                cxx = cxx,
                cxy = cxy,
                cyy = cyy,
            )

        FToB[findall(FToB == BC_JUMP_INTERFACE)] .= BC_LOCKED_INTERFACE

        nelems = length(rhsops)

        # initial conditions
        lenq0 = 2 * Np + 2 * Nq + 2 * Nq
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

            q[(lenq0 * (e - 1) + 1):(e * lenq0)] =
                [u0; v0; û10; û20; û30; û40]
        end

        if N == maximum(Nvals)
            mkpath("output")
            write_vtk(
                @sprintf(
                    "output/locked_N_blocks_sim_step_2p_%d_N0_%04d_%04d",
                    sbp_order,
                    N,
                    0
                ),
                metrics,
                q;
                cxx = cxx,
                cxy = cxy,
                cyy = cyy,
            )
        end

        hmin = mapreduce(m -> m.hmin, min, values(metrics))
        dt = cfl * hmin

        energy = zeros(length(ts))
        energy[1] = compute_energy(q, rhsops, Np)

        # Parameters to pass to the ODE solver
        params = (
            Nqr = Nq,
            Nqs = Nq,
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
            if N == maximum(Nvals)
                write_vtk(
                    @sprintf(
                        "output/locked_N_blocks_sim_step_2p_%d_N0_%04d_%04d",
                        sbp_order,
                        N,
                        step
                    ),
                    metrics,
                    q;
                    cxx = cxx,
                    cxy = cxy,
                    cyy = cyy,
                )
            end
            energy[step + 1] = compute_energy(q, rhsops, Np)
        end
        open(@sprintf("locked_energy_2p_%d_N0_%04d.csv", sbp_order, N), "w") do io
            writedlm(io, [Array(ts) sqrt.(energy / energy[1])])
        end

        data[N] = (q = q, rhsops = rhsops)
    end

    nelems = length(data[Nvals[1]].rhsops)
    ϵ = zeros(length(Nvals) - 1)

    for lvl in 1:(length(Nvals) - 1)
        N1 = Nvals[lvl + 1]
        Nq1 = N1 + 1
        Np1 = Nq1^2

        N2 = Nvals[lvl]
        Nq2 = N2 + 1
        Np2 = Nq2^2

        q1 = reshape(data[N1].q, Np1 + 2Nq1, 2, nelems)
        q2 = reshape(data[N2].q, Np2 + 2Nq2, 2, nelems)

        rhsops = data[N1].rhsops

        for e in 1:nelems
            u1 = reshape(q1[1:Np1, 1, e], Nq1, Nq1)
            u2 = reshape(q2[1:Np2, 1, e], Nq2, Nq2)
            Δu = u1 - u2[1:2:end, 1:2:end]

            Mv = rhsops[e].JH

            ϵ[lvl] += Δu[:]' * Mv * Δu[:]
        end
        ϵ[lvl] = sqrt(ϵ[lvl])
    end
    @show ϵ
    rate = (log.(ϵ[2:end]) - log.(ϵ[1:(end - 1)])) / log(2)
    @show rate
end
