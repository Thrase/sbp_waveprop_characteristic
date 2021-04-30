using SparseArrays
using LinearAlgebra
using SparseArrays

include("sim_funs.jl")
include("mms.jl")

let
    γ = 1
    β = 2

    T = Float64

    sbp_order = 4

    rho = 1 # TODO: account for non-unitary rho

    # mesh file side set type to actually boundary condition type
    bc_map =
        [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN, BC_JUMP_INTERFACE]
    (verts, EToV, EToF, FToB, EToDomain) =
        read_inp_2d("square_circle.inp"; bc_map = bc_map)

    # EToV defines the element by its vertices
    # EToF defines element by its four faces, in global face number
    # FToB defines whether face is Dirichlet (1), Neumann (2), frictional interface (7),
    # interior jump (8),  or just an interior interface (0)
    # EToDomain is 1 if element is inside circle; 2 otherwise

    # number of elements and faces
    (nelems, nfaces) = (size(EToV, 2), size(FToB, 1))
    @show (nelems, nfaces)

    # This is needed to fix up points that should be on the boundary of the
    # circle, but the mesh didn't quite put them there
    for v in 1:size(verts, 2)
        x, y = verts[1, v], verts[2, v]
        if abs(hypot(x, y) - 1) < 1e-5
            Q = atan(y, x)
            verts[1, v], verts[2, v] = cos(Q), sin(Q)
        end
    end

    # Plot the original connectivity before mesh warping
    # plot_connectivity(verts, EToV)

    # This is the base mesh size in each dimension on each element.
    N1 = N0 = 17

    no_refine = 3
    ϵ = zeros(no_refine)
    ϵ_exact = zeros(no_refine)

    # EToN0 is the base mesh size (e.g., before refinement)
    EToN0 = zeros(Int64, 2, nelems)
    EToN0[1, :] .= N0
    EToN0[2, :] .= N1

    #@assert typeof(EToV) == Array{Int, 2} && size(EToV) == (4, nelems)
    #@assert typeof(EToF) == Array{Int, 2} && size(EToF) == (4, nelems)
    #@assert maximum(maximum(EToF)) == nfaces

    # Determine secondary arrays
    (FToE, FToLF, EToO, EToS) = connectivityarrays(EToV, EToF)
    # FToE : Unique Global Face to Element Number
    #        (the i'th column of this stores the element numbers that share the
    #        global face number i)
    # FToLF: Unique Global Face to Element local face number
    #        (the i'th column of this stores the element local face numbers that
    #        shares the global face number i)
    # EToO : Element to Unique Global Faces Orientation
    #        (the i'th column of this stores the whether the element and global
    #        face are oriented in the same way in physical memory or need to be
    #        rotated)
    # EToS : Element to Unique Global Face Side
    #        (the i'th column of this stores whether an element face is on the
    #        plus side or minus side of the global face)

    # Exact solution
    Lx = maximum(verts[1, :])
    Ly = maximum(abs.(verts[2, :]))

    function friction(V)
        return β * asinh(γ * V)
    end

    gDfun(x, y, t, e) = ue(x, y, t, EToDomain[e])
    gDdotfun(x, y, t, e) = ue_t(x, y, t, EToDomain[e])
    function gNfun(nx, ny, xf, yf, t, e)
        dom = EToDomain[e]
        return nx .* (ue_x(xf, yf, t, dom)) + ny .* (ue_y(xf, yf, t, dom))
    end
    body_force(x, y, t, e) = force(x, y, t, EToDomain[e])

    for lvl in 1:length(ϵ)
        # Set up the local grid dimensions
        Nr = EToN0[1, :] * (2^(lvl - 1))
        Ns = EToN0[2, :] * (2^(lvl - 1))

        Nqr = Nr[1] + 1
        Nqs = Ns[1] + 1

        Np = Nqr * Nqs

        # Dictionary to store the operators (independent of element/block)
        mets = create_metrics(
            sbp_order,
            11,
            11,
            (r, s) -> (r, ones(size(r)), zeros(size(s))),
            (r, s) -> (s, zeros(size(r)), ones(size(s))),
        )

        OPTYPE = typeof(
            rhsoperators(
                rho,
                sbp_order,
                12,
                12,
                mets,
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
            ),
        )

        METTYPE = typeof(mets)
        rhsops = Dict{Int64, OPTYPE}()
        metrics = Dict{Int64, METTYPE}()

        # Build the local operators for each block

        for e in 1:nelems

            # Get domain corners
            (x1, x2, x3, x4) = verts[1, EToV[:, e]]
            (y1, y2, y3, y4) = verts[2, EToV[:, e]]

            # Initialize the domain transformations as transfinite between the corners
            ex = [
                (α) -> x1 * (1 .- α) / 2 + x3 * (1 .+ α) / 2,
                (α) -> x2 * (1 .- α) / 2 + x4 * (1 .+ α) / 2,
                (α) -> x1 * (1 .- α) / 2 + x2 * (1 .+ α) / 2,
                (α) -> x3 * (1 .- α) / 2 + x4 * (1 .+ α) / 2,
            ]
            exα = [
                (α) -> -x1 / 2 + x3 / 2,
                (α) -> -x2 / 2 + x4 / 2,
                (α) -> -x1 / 2 + x2 / 2,
                (α) -> -x3 / 2 + x4 / 2,
            ]
            ey = [
                (α) -> y1 * (1 .- α) / 2 + y3 * (1 .+ α) / 2,
                (α) -> y2 * (1 .- α) / 2 + y4 * (1 .+ α) / 2,
                (α) -> y1 * (1 .- α) / 2 + y2 * (1 .+ α) / 2,
                (α) -> y3 * (1 .- α) / 2 + y4 * (1 .+ α) / 2,
            ]
            eyα = [
                (α) -> -y1 / 2 + y3 / 2,
                (α) -> -y2 / 2 + y4 / 2,
                (α) -> -y1 / 2 + y2 / 2,
                (α) -> -y3 / 2 + y4 / 2,
            ]

            # For blocks on the circle, put in the curved edge transform
            if FToB[EToF[1, e]] == BC_JUMP_INTERFACE
                error("curved face 1 not implemented yet")
            end
            if FToB[EToF[2, e]] == BC_JUMP_INTERFACE
                error("curved face 2 not implemented yet")
            end
            if FToB[EToF[3, e]] == BC_JUMP_INTERFACE
                Q1 = atan(y1, x1)
                Q2 = atan(y2, x2)
                if !(-π / 2 < Q1 - Q2 < π / 2)
                    Q2 -= sign(Q2) * 2 * π
                end
                ex[3] = (α) -> cos.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
                ey[3] = (α) -> sin.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
                β3 = (Q2 - Q1) / 2
                exα[3] =
                    (α) -> -β3 .* sin.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
                eyα[3] =
                    (α) -> +β3 .* cos.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
            end
            if FToB[EToF[4, e]] == BC_JUMP_INTERFACE
                Q3 = atan(y3, x3)
                Q4 = atan(y4, x4)
                if !(-π / 2 < Q3 - Q4 < π / 2)
                    error("curved face 4 angle correction not implemented yet")
                end
                ex[4] = (α) -> cos.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
                ey[4] = (α) -> sin.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
                β4 = (Q4 - Q3) / 2
                exα[4] =
                    (α) -> -β4 .* sin.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
                eyα[4] =
                    (α) -> +β4 .* cos.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
            end

            # Create the volume transform as the transfinite blending of the edge
            # transformations
            xt(r, s) = transfinite_blend(
                ex[1],
                ex[2],
                ex[3],
                ex[4],
                exα[1],
                exα[2],
                exα[3],
                exα[4],
                r,
                s,
            )
            yt(r, s) = transfinite_blend(
                ey[1],
                ey[2],
                ey[3],
                ey[4],
                eyα[1],
                eyα[2],
                eyα[3],
                eyα[4],
                r,
                s,
            )

            metrics[e] = create_metrics(sbp_order, Nqr - 1, Nqs - 1, xt, yt)

            # Linear operators:
            rhsops[e] = rhsoperators(
                rho,
                sbp_order,
                Nqr,
                Nqs,
                metrics[e],
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
            )
        end

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
        tspan = (T(0), T(1))
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

        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:Np]
            #qe_v = qe[Np .+ (1:Np)]
            #qe_û1 = qe[2Np+1:2Np+Nqs]
            #qe_û2 = qe[2Np+1+Nqs:2(Np+Nqs)]
            #qe_û3 = qe[2(Np+Nqs)+1:2(Np+Nqs)+Nqr]
            #qe_û4 = qe[2(Np+Nqs)+Nqr+1:2(Np+Nqs+Nqr)]

            #τ̂ = (rhsops[e].nCB[1]*qe_u + 1*rhsops[e].nCnΓ[1]*qe_û1 - 1*rhsops[e].nCnΓL[1]*qe_u,
            #      rhsops[e].nCB[2]*qe_u + 1*rhsops[e].nCnΓ[2]*qe_û2 - 1*rhsops[e].nCnΓL[2]*qe_u,
            #      rhsops[e].nCB[3]*qe_u + 1*rhsops[e].nCnΓ[3]*qe_û3 - 1*rhsops[e].nCnΓL[3]*qe_u,
            #      rhsops[e].nCB[4]*qe_u + 1*rhsops[e].nCnΓ[4]*qe_û4 - 1*rhsops[e].nCnΓL[4]*qe_u
            #      )

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

            #τ̂exact = (rhsops[e].gN(tspan[2], e)[1], rhsops[e].gN(tspan[2], e)[2], rhsops[e].gN(tspan[2], e)[3], rhsops[e].gN(tspan[2], e)[4])

            Δu = qe_u - qexact_u
            Mv = rhsops[e].JH

            ϵ[lvl] += Δu' * Mv * Δu
        end # end compute error at lvl

        ϵ[lvl] = sqrt(ϵ[lvl])#/sqrt(ϵ_exact[lvl])
        @show (lvl, ϵ[lvl])
    end #loop over levels

    println((log.(ϵ[1:(end - 1)]) - log.(ϵ[2:end])) / log(2))
end
