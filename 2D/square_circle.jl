function build_square_circle(
    sbp_order,
    Nqr,
    Nqs,
    gDfun,
    gDdotfun,
    gNfun,
    body_force,
    bc_map,
    do_output = false;
    rho = 1,
    cxx = nothing,
    cxy = nothing,
    cyy = nothing,
)
    @assert Nqr == Nqs

    # EToV defines the element by its vertices
    # EToF defines element by its four faces, in global face number
    # FToB defines whether face is Dirichlet (1), Neumann (2), frictional interface (7),
    # interior jump (8),  or just an interior interface (0)
    # EToDomain is 1 if element is inside circle; 2 otherwise
    (verts, EToV, EToF, FToB, EToDomain) =
        read_inp_2d("square_circle.inp"; bc_map = bc_map)

    # Plot the original connectivity before mesh warping
    # plot_connectivity(verts, EToV)

    # This is needed to fix up points that should be on the boundary of the
    # circle, but the mesh didn't quite put them there
    for v in 1:size(verts, 2)
        x, y = verts[1, v], verts[2, v]
        if abs(hypot(x, y) - 1) < 1e-5
            Q = atan(y, x)
            verts[1, v], verts[2, v] = cos(Q), sin(Q)
        end
    end

    # Determine secondary arrays
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
    (FToE, FToLF, EToO, EToS) = connectivityarrays(EToV, EToF)

    # number of elements and faces
    (nelems, nfaces) = (size(EToV, 2), size(FToB, 1))
    if do_output
        @show (nelems, nfaces)
    end

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
            sbp_order,
            12,
            12,
            mets,
            gDfun,
            gDdotfun,
            gNfun,
            body_force;
            rho = rho,
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
        if abs(FToB[EToF[1, e]]) == BC_JUMP_INTERFACE
            error("curved face 1 not implemented yet")
        end
        if abs(FToB[EToF[2, e]]) == BC_JUMP_INTERFACE
            error("curved face 2 not implemented yet")
        end
        if abs(FToB[EToF[3, e]]) == BC_JUMP_INTERFACE
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
        if abs(FToB[EToF[4, e]]) == BC_JUMP_INTERFACE
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

        metrics[e] = create_metrics(sbp_order, Nqr - 1, Nqs - 1, xt, yt)

        if isnothing(cxx) && isnothing(cyy) && isnothing(cyy)
            crr = metrics[e].crr
            crs = metrics[e].crs
            css = metrics[e].css
        else
            J = metrics[e].J
            (x, y) = metrics[e].coord
            rx = metrics[e].rx
            ry = metrics[e].ry
            sx = metrics[e].sx
            sy = metrics[e].sy
            cxx_ = cxx.(x, y)
            cyy_ = cyy.(x, y)
            cyx_ = cxy_ = cxy.(x, y)
            crr =
                J .* (
                    rx .* cxx_ .* rx +
                    rx .* cxy_ .* ry +
                    ry .* cyx_ .* rx +
                    ry .* cyy_ .* ry
                )
            crs =
                J .* (
                    rx .* cxx_ .* sx +
                    rx .* cxy_ .* sy +
                    ry .* cyx_ .* sx +
                    ry .* cyy_ .* sy
                )
            css =
                J .* (
                    sx .* cxx_ .* sx +
                    sx .* cxy_ .* sy +
                    sy .* cyx_ .* sx +
                    sy .* cyy_ .* sy
                )
        end

        # Linear operators:
        rhsops[e] = rhsoperators(
            sbp_order,
            Nqr,
            Nqs,
            metrics[e],
            gDfun,
            gDdotfun,
            gNfun,
            body_force;
            rho = rho,
            crr = crr,
            crs = crs,
            css = css,
        )
    end

    do_output && plot_blocks(metrics, abs.(FToB), EToF)

    return (
        metrics = metrics,
        rhsops = rhsops,
        EToDomain = EToDomain,
        EToF = EToF,
        EToO = EToO,
        EToS = EToS,
        FToB = FToB,
        FToE = FToE,
        FToLF = FToLF,
    )
end
