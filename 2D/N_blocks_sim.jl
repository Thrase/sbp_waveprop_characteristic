using SparseArrays
using LinearAlgebra
using SparseArrays
using Printf

include("sim_funs.jl")

#import PGFPlots
γ = 1
β = 2

T = Float64

p = 2

rho = 1  # TODO: account for non-unitary rho

# mesh file side set type to actually boundary condition type
bc_map = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN, BC_JUMP_INTERFACE]
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
N1 = N0 = 48

no_refine = 1
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
μ1 = 0.1
μ2 = 0.2
σ1 = 0.005
σ2 = 0.005

#Set initial displacement (just function of x and y)
vinside(x, y) = begin
    return exp.(-(x .- μ1) .^ 2 ./ σ1 .- (y .- μ2) .^ 2 ./ 2σ2)
end
voutside(x, y) = begin
    return exp.(-(x .- μ1) .^ 2 ./ σ1 .- (y .- μ2) .^ 2 ./ 2σ2)
end
ue(x, y, e) = begin
    if EToDomain[e] == 1
        return vinside(x, y)
    elseif EToDomain[e] == 2
        return voutside(x, y)
    else
        error("invalid block")
    end
end

#Set initial velocity (just function of x and y)
vinside_t(x, y) = begin
    r = sqrt.(x .^ 2 + y .^ 2)
    return 0 .* x
end
voutside_t(x, y) = begin
    r = sqrt.(x .^ 2 + y .^ 2)
    return 0 .* x
end
ue_t(x, y, e) = begin
    if EToDomain[e] == 1
        return vinside_t(x, y)
    elseif EToDomain[e] == 2
        return voutside_t(x, y)
    else
        error("invalid block")
    end
end

#Set dirichlet data at left/right boundaries
gDfun(x, y, t, e) = begin
    return 0 .* x
end

gDdotfun(x, y, t, e) = begin
    return 0 .* x
end

#Set neumann data at top/bottom boundaries
gNfun(nx, ny, x, y, t, e) = begin
    return 0 .* x
end

# Friction Law at fault interfacd
Friction(V) = begin
    return β * asinh(γ * V)
end

for lvl in 1:1#length(ϵ)
    # Set up the local grid dimensions
    Nr = EToN0[1, :] * (2^(lvl - 1))
    Ns = EToN0[2, :] * (2^(lvl - 1))

    Nqr = Nr[1] + 1
    Nqs = Ns[1] + 1

    N = Nqr * Nqs

    # Dictionary to store the operators (independent of element/block)
    xt = (r, s) -> (r, ones(size(r)), zeros(size(s)))
    yt = (r, s) -> (s, zeros(size(r)), ones(size(s)))
    mets = create_metrics(p, 11, 11, xt, yt)

    #ddddds
    OPTYPE = typeof(rhsoperators(rho, p, 12, 12, mets, gDfun, gDdotfun, gNfun))

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
            exα[3] = (α) -> -β3 .* sin.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
            eyα[3] = (α) -> +β3 .* cos.(Q1 * (1 .- α) / 2 + Q2 * (1 .+ α) / 2)
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
            exα[4] = (α) -> -β4 .* sin.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
            eyα[4] = (α) -> +β4 .* cos.(Q3 * (1 .- α) / 2 + Q4 * (1 .+ α) / 2)
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
        rhsops[e] =
            rhsoperators(rho, p, Nqr, Nqs, metrics[e], gDfun, gDdotfun, gNfun)
    end

    # ODE RHS function
    function waveprop!(dq, q, p, t)

        #@show t

        @assert length(q) == nelems * (2 * N + 2 * Nqr + 2 * Nqs)

        for e in 1:nelems
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

                    qplus = @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                    uplus = @view qplus[1:N]
                    vplus = @view qplus[N .+ (1:N)]
                    û1plus = qplus[(2N + 1):(2N + Nqs)]
                    û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                    û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                    û4plus = qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

                    ustarplus = (û1plus, û2plus, û3plus, û4plus)
                    τ̂plus =
                        rhsops[cel[1]].nCB[cfc[1]] * uplus +
                        rhsops[cel[1]].nCnΓ[cfc[1]] * ustarplus[cfc[1]] -
                        rhsops[cel[1]].nCnΓL[cfc[1]] * uplus

                    revrse_e = EToO[lf, e]
                    revrse_cel = EToO[cfc, cel]

                    vplus_fc = rhsops[cel[1]].L[cfc[1]] * vplus
                    Zplus = rhsops[cel[1]].Z[cfc[1]]

                    sJplus = rhsops[cel[1]].sJ[cfc[1]]
                    sJminus = rhsops[e].sJ[lf]

                    if EToO[lf, e] != EToO[cfc[1], cel[1]]
                        τ̂plus_rev = τ̂plus[end:-1:1]
                        vplus_fc_rev = vplus_fc[end:-1:1]
                        Z_rev = Zplus[end:-1:1, end:-1:1]
                        sJplus_rev = sJplus[end:-1:1]
                    else
                        τ̂plus_rev = τ̂plus
                        vplus_fc_rev = vplus_fc
                        Z_rev = Zplus
                        sJplus_rev = sJplus
                    end

                    Z = rhsops[e].Z[lf]

                    #Vminus = gDdotplus_rev - rhsops[e].gDdot(t, e)[lf]

                    Zvminus = Z * rhsops[e].L[lf] * v
                    Zvplus = Z_rev * vplus_fc_rev

                    vfric = zeros(Float64, Nqr)

                    for i in 1:Nqr
                        f(V) =
                            2 * sJminus[i] * Friction(V) + Z[i, i] * V -
                            (Zvplus[i] - Zvminus[i]) + τ̂plus_rev[i] - τ̂[lf][i]

                        Left =
                            (1 / Z[i, i]) * (
                                -τ̂plus_rev[i] + τ̂[lf][i] + Zvplus[i] -
                                Zvminus[i]
                            )

                        Right = -Left

                        if Left >= Right
                            tmp = Left
                            Left = Right
                            Right = tmp
                        end

                        v_root = brentdekker(f, Left, Right)
                        vfric[i] = v_root[1]
                    end

                    vstar[lf] .=
                        rhsops[e].L[lf] * v +
                        rhsops[e].Zinv[lf] *
                        (-1 * τ̂[lf] + rhsops[e].sJ[lf] .* Friction.(vfric))
                    τ̂star[lf] .=
                        rhsops[e].Z[lf] * (vstar[lf] - rhsops[e].L[lf] * v) +
                        τ̂[lf]

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

                    qplus = @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                    uplus = @view qplus[1:N]
                    vplus = @view qplus[N .+ (1:N)]
                    û1plus = qplus[(2N + 1):(2N + Nqs)]
                    û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                    û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                    û4plus = qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

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

                    qplus = @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                    uplus = @view qplus[1:N]
                    vplus = @view qplus[N .+ (1:N)]
                    û1plus = qplus[(2N + 1):(2N + Nqs)]
                    û2plus = qplus[(2N + 1 + Nqs):(2(N + Nqs))]
                    û3plus = qplus[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
                    û4plus = qplus[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

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

            dv .= rhsops[e].JIHI * dv

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

        u0 = ue.(metrics[e].coord[1][:], metrics[e].coord[2][:], e)
        v0 = ue_t.(metrics[e].coord[1][:], metrics[e].coord[2][:], e)
        û10 = ue.(xf1[:], yf1[:], e)
        û20 = ue.(xf2[:], yf2[:], e)
        û30 = ue.(xf3[:], yf3[:], e)
        û40 = ue.(xf4[:], yf4[:], e)

        q[(lenq0 * (e - 1) + 1):(e * lenq0)] = [u0; v0; û10; û20; û30; û40]
    end

    mkpath("output")
    write_vtk(@sprintf("output/N_blocks_sim_step_%04d", 0), metrics, q)

    hmin = mapreduce(m -> m.hmin, min, values(metrics))
    dt = 2hmin

    # solve the ODE
    ts = 0:0.1:5
    energy = zeros(length(ts))
    for step in 1:(length(ts) - 1)
        tspan = (ts[step], ts[step + 1])
        timestep!(q, waveprop!, nothing, dt, tspan)
        write_vtk(@sprintf("output/N_blocks_sim_step_%04d", step), metrics, q)

        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:N]
            qe_v = qe[N .+ (1:N)]
            qe_û1 = qe[(2N + 1):(2N + Nqs)]
            qe_û2 = qe[(2N + 1 + Nqs):(2(N + Nqs))]
            qe_û3 = qe[(2(N + Nqs) + 1):(2(N + Nqs) + Nqr)]
            qe_û4 = qe[(2(N + Nqs) + Nqr + 1):(2(N + Nqs + Nqr))]

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
