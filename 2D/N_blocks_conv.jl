using SparseArrays
using LinearAlgebra
using SparseArrays

include("sim_funs.jl")

function vinside(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return sin.(t) .* (1 .- exp.(-1 .* r .^ 2)) .* r .* sin.(theta)
end

function vinside_t(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return cos.(t) .* (1 .- exp.(-1 .* r .^ 2)) .* r .* sin.(theta)
end

function vinside_tt(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return -sin.(t) .* (1 .- exp.(-1 .* r .^ 2)) .* r .* sin.(theta)
end

function vinside_x(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    dtheta_dx = -1 .* sin.(theta) ./ r
    dr_dx = cos.(theta)
    dv_dr =
        (2 .* r .^ 2 .* exp.(-1 .* r .^ 2) .+ 1 .- exp.(-1 .* r .^ 2)) .*
        sin.(theta)
    dv_dtheta = (1 .- exp.(-1 .* r .^ 2)) .* r .* cos.(theta)
    return sin.(t) .* (dv_dr .* dr_dx + dv_dtheta .* dtheta_dx)
end

function vinside_y(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    dtheta_dy = cos.(theta) ./ r
    dr_dy = sin.(theta)
    dv_dr =
        (2 .* r .^ 2 .* exp.(-1 .* r .^ 2) .+ 1 .- exp.(-1 .* r .^ 2)) .*
        sin.(theta)
    dv_dtheta = (1 .- exp.(-1 .* r .^ 2)) .* r .* cos.(theta)
    return sin.(t) .* (dv_dr .* dr_dy + dv_dtheta .* dtheta_dy)
end

function force_inside(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    u_r =
        (2 .* r .^ 2 .* exp.(-1 .* r .^ 2) .+ 1 .- exp.(-1 .* r .^ 2)) .*
        sin.(theta)
    u_rr = exp.(-1 .* r .^ 2) .* (6 .* r .- 4 .* r .^ 3) .* sin.(theta)
    u_thetatheta = -(1 .- exp.(-1 .* r .^ 2)) .* r .* sin.(theta)
    return vinside_tt(x, y, t) .-
           sin.(t) * (u_rr .+ (1 ./ r) .* u_r .+ (1 ./ r .^ 2) .* u_thetatheta)
end

function voutside(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return sin.(t) .* ((r .- 1) .^ 2 .* cos.(theta) .+ (r .- 1) .* sin.(theta))
end

function voutside_t(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return cos.(t) .* ((r .- 1) .^ 2 .* cos.(theta) .+ (r .- 1) .* sin.(theta))
end

function voutside_tt(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    return -sin.(t) .* ((r .- 1) .^ 2 .* cos.(theta) .+ (r .- 1) .* sin.(theta))
end

function voutside_x(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    dtheta_dx = -1 .* sin.(theta) ./ r
    dr_dx = cos.(theta)
    dv_dr = 2 .* (r .- 1) .* cos.(theta) .+ sin.(theta)
    dv_dtheta = -1 .* (r .- 1) .^ 2 .* sin.(theta) .+ (r .- 1) .* cos.(theta)
    return sin.(t) .* (dv_dr .* dr_dx + dv_dtheta .* dtheta_dx)
end

function voutside_y(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    dtheta_dy = cos.(theta) ./ r
    dr_dy = sin.(theta)
    dv_dr = 2 .* (r .- 1) .* cos.(theta) .+ sin.(theta)
    dv_dtheta = -1 .* (r .- 1) .^ 2 .* sin.(theta) .+ (r .- 1) .* cos.(theta)
    return sin.(t) .* (dv_dr .* dr_dy + dv_dtheta .* dtheta_dy)
end

function force_outside(x, y, t)
    r = sqrt.(x .^ 2 + y .^ 2)
    theta = atan.(y, x)
    u_thetatheta = -1 .* (r .- 1) .^ 2 .* cos.(theta) .- (r .- 1) .* sin.(theta)
    u_r = 2 .* (r .- 1) .* cos.(theta) .+ sin.(theta)
    u_rr = 2 * cos.(theta)
    return voutside_tt(x, y, t) .-
           sin.(t) * (u_rr .+ (1 ./ r) .* u_r .+ (1 ./ r .^ 2) .* u_thetatheta)
end

function ue(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
    if dom == 1
        return A1 * vinside(x, y, t)
    elseif dom == 2
        return A2 * voutside(x, y, t)
    else
        error("invalid block")
    end
end

function ue_x(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
    if dom == 1
        return A1 * vinside_x(x, y, t)
    elseif dom == 2
        return A2 * voutside_x(x, y, t)
    else
        error("invalid block")
    end
end

function ue_y(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
    if dom == 1
        return A1 * vinside_y(x, y, t)
    elseif dom == 2
        return A2 * voutside_y(x, y, t)
    else
        error("invalid block")
    end
end

function ue_t(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
    if dom == 1
        return A1 * vinside_t(x, y, t)
    elseif dom == 2
        return A2 * voutside_t(x, y, t)
    else
        error("invalid block")
    end
end

#u_rr + (1/r)*u_r + (1/r^2)*u_theta,theta
function force(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
    if dom == 1
        return A1 * force_inside(x, y, t)
    elseif dom == 2
        return A2 * force_outside(x, y, t)
    else
        error("invalid block")
    end
end

let
    γ = 1
    β = 2

    T = Float64

    p = 4

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

    #=pgf_axis = PGFPlots.Axis(style="width=5cm, height=5cm, ticks=none",
                             xlabel=PGFPlots.L"$x$",
                             ylabel=PGFPlots.L"$y$",
                             xmin = -2, xmax = 2,
                             ymin = -2, ymax = 2)
    for f in 1:nfaces
      if FToB[f] != BC_JUMP_INTERFACE
        (e, lf) = FToE[1,f], FToLF[1,f]
        if lf == 1
          v1, v2 = EToV[1, e], EToV[3, e]
        elseif lf == 2
          v1, v2 = EToV[2, e], EToV[4, e]
        elseif lf == 3
          v1, v2 = EToV[1, e], EToV[2, e]
        else
          v1, v2 = EToV[3, e], EToV[4, e]
        end
        x = verts[1, [v1 v2]][:]
        y = verts[2, [v1 v2]][:]
        push!(pgf_axis, PGFPlots.Linear(x, y, style="no marks, solid, black"))
      end
    end
    push!(pgf_axis, PGFPlots.Circle(0,0,1, style = "very thick, red"))
    PGFPlots.save("square_circle.tikz", pgf_axis)
    =#
    # Exact solution
    Lx = maximum(verts[1, :])
    Ly = maximum(abs.(verts[2, :]))

    function Friction(V)
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

        N = Nqr * Nqs

        # Dictionary to store the operators (independent of element/block)
        mets = create_metrics(
            p,
            11,
            11,
            (r, s) -> (r, ones(size(r)), zeros(size(s))),
            (r, s) -> (s, zeros(size(r)), ones(size(s))),
        )

        OPTYPE = typeof(
            rhsoperators(
                rho,
                p,
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

            metrics[e] = create_metrics(p, Nqr - 1, Nqs - 1, xt, yt)

            # Linear operators:
            rhsops[e] = rhsoperators(
                rho,
                p,
                Nqr,
                Nqs,
                metrics[e],
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
            )
        end

        # ODE RHS function
        function waveprop!(dq, q, p, t)

            # @show t

            @assert length(q) == nelems * (2 * N + 2 * Nqr + 2 * Nqs)

            @inbounds for e in 1:nelems
                qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]
                dqe = @view dq[(lenq0 * (e - 1) + 1):(e * lenq0)]

                u = qe[1:N]
                v = qe[N .+ (1:N)]
                û1 = qe[(2N + 1):(2N + Nqs)]
                û2 = qe[(2N + 1 + Nqs):(2(N + Nqs))]
                û3 = qe[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                û4 = qe[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                ustar = (û1, û2, û3, û4)

                du = @view dqe[1:N]
                dv = @view dqe[N .+ (1:N)]
                dû1 = @view dqe[(2N + 1):(2N + Nqs)]
                dû2 = @view dqe[(2N + 1 + Nqs):(2(N + Nqs))]
                dû3 = @view dqe[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                dû4 = @view dqe[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                #vstar = (dû1, dû2, dû3, dû4)

                τ̂ = (
                    rhsops[e].nCB[1] * u + rhsops[e].nCnΓ[1] * û1 -
                    rhsops[e].nCnΓL[1] * u,
                    rhsops[e].nCB[2] * u + rhsops[e].nCnΓ[2] * û2 -
                    rhsops[e].nCnΓL[2] * u,
                    rhsops[e].nCB[3] * u + rhsops[e].nCnΓ[3] * û3 -
                    rhsops[e].nCnΓL[3] * u,
                    rhsops[e].nCB[4] * u + rhsops[e].nCnΓ[4] * û4 -
                    rhsops[e].nCnΓL[4] * u,
                )

                τ̂star = (
                    zeros(Float64, Nqs),
                    zeros(Float64, Nqs),
                    zeros(Float64, Nqr),
                    zeros(Float64, Nqr),
                )

                vstar = (
                    zeros(Float64, Nqs),
                    zeros(Float64, Nqs),
                    zeros(Float64, Nqr),
                    zeros(Float64, Nqr),
                )

                glb_fcs = EToF[:, e]  #global faces of element e

                for lf in 1:4
                    bc_type_face = FToB[glb_fcs[lf]]
                    if bc_type_face == 1
                        τ̂star[lf] .= τ̂[lf]
                        vstar[lf] .= rhsops[e].gDdot(t, e)[lf]
                        ustar[lf] .= rhsops[e].gD(t, e)[lf]

                    elseif bc_type_face == 2 #Neumann
                        τ̂star[lf] .= rhsops[e].gN(t, e)[lf]
                        vstar[lf] .= rhsops[e].L[lf] * v
                        ustar[lf] .= rhsops[e].L[lf] * u

                    elseif bc_type_face == 7
                        els_share = FToE[:, glb_fcs[lf]]
                        lf_share_glb = FToLF[:, glb_fcs[lf]]
                        cel = [1]
                        cfc = [1]
                        for i in 1:2
                            if els_share[i] != e
                                cel[1] = els_share[i] # other element face is connected to!
                            end
                        end

                        if e < cel[1]
                            cfc[1] = lf_share_glb[2]
                        else
                            cfc[1] = lf_share_glb[1]
                        end

                        qplus =
                            @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                        uplus = @view qplus[1:N]
                        vplus = @view qplus[N .+ (1:N)]
                        û1plus = qplus[(2N + 1):(2N + Nqs)]
                        û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                        û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                        û4plus =
                            qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                        ustarplus = (û1plus, û2plus, û3plus, û4plus)
                        τ̂plus =
                            rhsops[cel[1]].nCB[cfc[1]] * uplus +
                            rhsops[cel[1]].nCnΓ[cfc[1]] * ustarplus[cfc[1]] -
                            rhsops[cel[1]].nCnΓL[cfc[1]] * uplus

                        revrse_e = EToO[lf, e]
                        revrse_cel = EToO[cfc, cel]

                        vplus_fc = rhsops[cel[1]].L[cfc[1]] * vplus
                        Zplus = rhsops[cel[1]].Z[cfc[1]]

                        gDdotplus = rhsops[cel[1]].gDdot(t, cel[1])[cfc[1]]
                        gNplus = rhsops[cel[1]].gN(t, cel[1])[cfc[1]]
                        sJplus = rhsops[cel[1]].sJ[cfc[1]]
                        sJminus = rhsops[e].sJ[lf]

                        if EToO[lf, e] != EToO[cfc[1], cel[1]]
                            τ̂plus_rev = τ̂plus[end:-1:1]
                            vplus_fc_rev = vplus_fc[end:-1:1] #TODO: could change these and above to avoid defining new arrays
                            Z_rev = Zplus[end:-1:1, end:-1:1]
                            gNplus_rev = gNplus[end:-1:1]
                            gDdotplus_rev = gDdotplus[end:-1:1]
                            sJplus_rev = sJplus[end:-1:1]
                        else
                            τ̂plus_rev = τ̂plus
                            vplus_fc_rev = vplus_fc
                            Z_rev = Zplus
                            gNplus_rev = gNplus
                            gDdotplus_rev = gDdotplus
                            sJplus_rev = sJplus
                        end

                        # Z_refI = spdiagm(0 => 1 ./ diag(Z_rev))
                        # ZI = rhsops[e].Zinv[lf]
                        Z = rhsops[e].Z[lf]

                        Vminus = gDdotplus_rev - rhsops[e].gDdot(t, e)[lf]
                        sJgminus =
                            rhsops[e].gN(t, e)[lf] -
                            rhsops[e].sJ[lf] .* Friction.(Vminus)
                        sJgplus = gNplus_rev - sJplus_rev .* Friction.(-Vminus)

                        #gminus = rhsops[e].Zinv[lf] * sJgminus
                        #gplus = Z_refI * sJgplus

                        Zvminus = Z * rhsops[e].L[lf] * v
                        Zvplus = Z_rev * vplus_fc_rev

                        vfric = zeros(Float64, Nqr)

                        for i in 1:Nqr
                            f(V) =
                                2 * sJminus[i] * Friction(V) + Z[i, i] * V -
                                (Zvplus[i] - Zvminus[i]) + τ̂plus_rev[i] -
                                τ̂[lf][i] - sJgplus[i] + sJgminus[i]

                            Left =
                                (1 / Z[i, i]) * (
                                    sJgplus[i] - sJgminus[i] - τ̂plus_rev[i] +
                                    τ̂[lf][i] +
                                    Zvplus[i] - Zvminus[i]
                                )

                            Right = -Left

                            if Left >= Right
                                tmp = Left
                                Left = Right
                                Right = tmp
                            end

                            v_root = brentdekker(f, Left, Right)#, xatol = 1e-9, xrtol = 1e-9, ftol = 1e-9)
                            vfric[i] = v_root[1]
                        end

                        vstar[lf] .=
                            rhsops[e].L[lf] * v +
                            rhsops[e].Zinv[lf] * (
                                -1 * τ̂[lf] +
                                rhsops[e].sJ[lf] .* Friction.(vfric) +
                                sJgminus
                            )
                        τ̂star[lf] .=
                            rhsops[e].Z[lf] *
                            (vstar[lf] - rhsops[e].L[lf] * v) + τ̂[lf]

                    elseif bc_type_face == 8 #interior jump
                        els_share = FToE[:, glb_fcs[lf]]
                        lf_share_glb = FToLF[:, glb_fcs[lf]]
                        cel = [1]
                        cfc = [1]
                        for i in 1:2
                            if els_share[i] != e
                                cel[1] = els_share[i] # other element face is connected to!
                            end
                        end

                        if e < cel[1]
                            cfc[1] = lf_share_glb[2]
                        else
                            cfc[1] = lf_share_glb[1]
                        end

                        qplus =
                            @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                        uplus = @view qplus[1:N]
                        vplus = @view qplus[N .+ (1:N)]
                        û1plus = qplus[(2N + 1):(2N + Nqs)]
                        û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                        û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                        û4plus =
                            qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                        ustarplus = (û1plus, û2plus, û3plus, û4plus)
                        τ̂plus =
                            rhsops[cel[1]].nCB[cfc[1]] * uplus +
                            rhsops[cel[1]].nCnΓ[cfc[1]] * ustarplus[cfc[1]] -
                            rhsops[cel[1]].nCnΓL[cfc[1]] * uplus

                        revrse_e = EToO[lf, e]
                        revrse_cel = EToO[cfc, cel]
                        vplus_fc = rhsops[cel[1]].L[cfc[1]] * vplus

                        dataplus_dot = rhsops[cel[1]].gDdot(t, cel[1])[cfc[1]]

                        if EToO[lf, e] != EToO[cfc[1], cel[1]]
                            τ̂plus_rev = τ̂plus[end:-1:1]
                            vplus_fc_rev = vplus_fc[end:-1:1] #TODO: could change these and above to avoid defining new arrays
                            dataplus_dot .= dataplus_dot[end:-1:1]
                        else
                            τ̂plus_rev = τ̂plus
                            vplus_fc_rev = vplus_fc
                        end

                        τ̂star[lf] .= 0.5 * (τ̂[lf] - τ̂plus_rev)
                        delta_dot = rhsops[e].gDdot(t, e)[lf] - dataplus_dot

                        if EToS[lf, e] == 1 #on minus side
                            vstar[lf] .=
                                0.5 * (rhsops[e].L[lf] * v + vplus_fc_rev) +
                                0.5 * delta_dot
                        else
                            vstar[lf] .=
                                0.5 * (rhsops[e].L[lf] * v + vplus_fc_rev) +
                                0.5 * delta_dot
                        end

                    elseif bc_type_face == 0
                        els_share = FToE[:, glb_fcs[lf]]
                        lf_share_glb = FToLF[:, glb_fcs[lf]]
                        cel = [1]
                        cfc = [1]
                        for i in 1:2
                            if els_share[i] != e
                                cel[1] = els_share[i] # other element face is connected to!
                            end
                        end

                        if e < cel[1]
                            cfc[1] = lf_share_glb[2]
                        else
                            cfc[1] = lf_share_glb[1]
                        end

                        qplus =
                            @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                        uplus = @view qplus[1:N]
                        vplus = @view qplus[N .+ (1:N)]
                        û1plus = qplus[(2N + 1):(2N + Nqs)]
                        û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                        û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                        û4plus =
                            qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                        ustarplus = (û1plus, û2plus, û3plus, û4plus)
                        τ̂plus =
                            rhsops[cel[1]].nCB[cfc[1]] * uplus +
                            rhsops[cel[1]].nCnΓ[cfc[1]] * ustarplus[cfc[1]] -
                            rhsops[cel[1]].nCnΓL[cfc[1]] * uplus

                        revrse_e = EToO[lf, e]
                        revrse_cel = EToO[cfc, cel]
                        vplus_fc = rhsops[cel[1]].L[cfc[1]] * vplus

                        if EToO[lf, e] != EToO[cfc[1], cel[1]]#e == 1 || e == 2
                            τ̂plus_rev = τ̂plus[end:-1:1]
                            vplus_fc_rev = vplus_fc[end:-1:1]
                        else
                            τ̂plus_rev = τ̂plus
                            vplus_fc_rev = vplus_fc
                        end

                        τ̂star[lf] .= 0.5 * (τ̂[lf] - τ̂plus_rev)
                        vstar[lf] .= 0.5 * (rhsops[e].L[lf] * v + vplus_fc_rev)
                    else
                        print("illegal bc type")
                        return
                    end
                end

                du .= v
                dv .= -rhsops[e].Ã * u

                for lf in 1:4
                    dv .+=
                        rhsops[e].L[lf]' * rhsops[e].H[lf] * τ̂star[lf] -
                        rhsops[e].BtnCH[lf] * (ustar[lf] - rhsops[e].L[lf] * u)
                end

                dv .= rhsops[e].JIHI * dv .+ rhsops[e].F(t, e)

                dû1 .= vstar[1]
                dû2 .= vstar[2]
                dû3 .= vstar[3]
                dû4 .= vstar[4]
            end
        end

        # initial conditions
        lenq0 = 2 * N + 2 * Nqr + 2 * Nqs
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
        timestep!(q, waveprop!, nothing, dt, tspan)

        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:N]
            #qe_v = qe[N .+ (1:N)]
            #qe_û1 = qe[2N+1:2N+Nqs]
            #qe_û2 = qe[2N+1+Nqs:2(N+Nqs)]
            #qe_û3 = qe[2(N+Nqs)+1:2(N+Nqs)+Nqr]
            #qe_û4 = qe[2(N+Nqs)+Nqr+1:2(N+Nqs+Nqr)]

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
            #Δv = qe_v - qexact_v
            #Δû1 = qe_û1 - qexact_û1
            #Δû2 = qe_û2 - qexact_û2
            #Δû3 = qe_û3 - qexact_û3
            #Δû4 = qe_û4 - qexact_û4

            #Δτ1 = τ̂[1] - τ̂exact[1]
            #Δτ2 = τ̂[2] - τ̂exact[2]

            #Δτ3 = τ̂[3] - τ̂exact[3]
            #Δτ4 = τ̂[4] - τ̂exact[4]

            #Mu = rhsops[e].Ã
            Mv = rhsops[e].JH

            #=Mû1 = rhsops[e].H[1]
            Mû2 = rhsops[e].H[2]
            Mû3 = rhsops[e].H[3]
            Mû4 = rhsops[e].H[4]

            Mτ̂1 = rhsops[e].X[1] * rhsops[e].H[1]
            Mτ̂2 = rhsops[e].X[2] * rhsops[e].H[2]
            Mτ̂3 = rhsops[e].X[3] * rhsops[e].H[3]
            Mτ̂4 = rhsops[e].X[4] * rhsops[e].H[4]

            Mu1 = (rhsops[e].nCB[1])' * rhsops[e].X[1] * rhsops[e].H[1] * rhsops[e].nCB[1]
            Mu2 = (rhsops[e].nCB[2])' * rhsops[e].X[2] * rhsops[e].H[2] * rhsops[e].nCB[2]
            Mu3 = (rhsops[e].nCB[3])' * rhsops[e].X[3] * rhsops[e].H[3] * rhsops[e].nCB[3]
            Mu4 = (rhsops[e].nCB[4])' * rhsops[e].X[4] * rhsops[e].H[4] * rhsops[e].nCB[4]
            =#
            ϵ[lvl] += Δu' * Mv * Δu

            #ϵ_exact[lvl] += qexact_u' * Mv * qexact_u

            #=ϵ[lvl] += 0.5 * Δv' * Mv * Δv + 0.5 * Δu' * Mu * Δu +
                          0.5 * (1*Δτ1' * Mτ̂1 * Δτ1 - 0*Δu' * Mu1 * Δu +
                                 1*Δτ2' * Mτ̂2 * Δτ2 - 0*Δu' * Mu2 * Δu +
                                 1*Δτ3' * Mτ̂3 * Δτ3 - 0*Δu' * Mu3 * Δu +
                                1*Δτ4' * Mτ̂4 * Δτ4 - 0*Δu' * Mu4 * Δu)
            =#
            #ϵ[lvl] += sqrt( 0.5 * (Δτ1' * Mτ̂1 * Δτ1   + Δτ2' * Mτ̂2 * Δτ2  + Δτ3' * Mτ̂3 * Δτ3 + Δτ4' * Mτ̂4 * Δτ4 ))

        end # end compute error at lvl

        ϵ[lvl] = sqrt(ϵ[lvl])#/sqrt(ϵ_exact[lvl])
        @show (lvl, ϵ[lvl])
    end #loop over levels

    println((log.(ϵ[1:(end - 1)]) - log.(ϵ[2:end])) / log(2))
end
