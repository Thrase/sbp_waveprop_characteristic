using SparseArrays
using LinearAlgebra
using UnicodePlots
using WriteVTK
using DiagonalSBP

# flatten tuples to arrays
flatten_tuples(x) =
    reshape(collect(Iterators.flatten(x)), length(x[1]), length(x))

⊗(A, B) = kron(A, B)

const BC_DIRICHLET = 1
const BC_NEUMANN = 2
const BC_LOCKED_INTERFACE = 0
const BC_JUMP_INTERFACE = 7

# Transfinite Blend
function transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
    # +---4---+
    # |       |
    # 1       2
    # |       |
    # +---3---+
    @assert [α1(-1) α2(-1) α1(1) α2(1)] ≈ [α3(-1) α3(1) α4(-1) α4(1)]

    x =
        (1 .+ r) .* α2(s) / 2 +
        (1 .- r) .* α1(s) / 2 +
        (1 .+ s) .* α4(r) / 2 +
        (1 .- s) .* α3(r) / 2 -
        (
            (1 .+ r) .* (1 .+ s) .* α2(1) +
            (1 .- r) .* (1 .+ s) .* α1(1) +
            (1 .+ r) .* (1 .- s) .* α2(-1) +
            (1 .- r) .* (1 .- s) .* α1(-1)
        ) / 4

    xr =
        α2(s) / 2 - α1(s) / 2 +
        (1 .+ s) .* α4r(r) / 2 +
        (1 .- s) .* α3r(r) / 2 -
        (
            +(1 .+ s) .* α2(1) +
            -(1 .+ s) .* α1(1) +
            +(1 .- s) .* α2(-1) +
            -(1 .- s) .* α1(-1)
        ) / 4

    xs =
        (1 .+ r) .* α2s(s) / 2 + (1 .- r) .* α1s(s) / 2 + α4(r) / 2 -
        α3(r) / 2 -
        (
            +(1 .+ r) .* α2(1) +
            +(1 .- r) .* α1(1) +
            -(1 .+ r) .* α2(-1) +
            -(1 .- r) .* α1(-1)
        ) / 4

    return (x, xr, xs)
end

function transfinite_blend(α1, α2, α3, α4, r, s, p)
    (Nrp, Nsp) = size(r)
    (Dr, _, _, _) = DiagonalSBP.D1(p, Nrp - 1; xc = (-1, 1))
    (Ds, _, _, _) = DiagonalSBP.D1(p, Nsp - 1; xc = (-1, 1))

    α2s(s) = α2(s) * Ds'
    α1s(s) = α1(s) * Ds'
    α4r(s) = Dr * α4(r)
    α3r(s) = Dr * α3(r)

    transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
end

function transfinite_blend(
    v1::T1,
    v2::T2,
    v3::T3,
    v4::T4,
    r,
    s,
) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}
    e1(α) = v1 * (1 .- α) / 2 + v3 * (1 .+ α) / 2
    e2(α) = v2 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
    e3(α) = v1 * (1 .- α) / 2 + v2 * (1 .+ α) / 2
    e4(α) = v3 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
    e1α(α) = -v1 / 2 + v3 / 2
    e2α(α) = -v2 / 2 + v4 / 2
    e3α(α) = -v1 / 2 + v2 / 2
    e4α(α) = -v3 / 2 + v4 / 2
    transfinite_blend(e1, e2, e3, e4, e1α, e2α, e3α, e4α, r, s)
end

# connectivityarrays
function connectivityarrays(EToV, EToF)
    # number of elements
    nelems = size(EToV, 2)
    nfaces = maximum(maximum(EToF))

    # Determine secondary arrays
    # FToE : Unique Global Face to Element Number
    # FToLF: Unique Global Face to Element local face number
    # EToO : Element to Unique Global Faces Orientation
    # EToS : Element to Unique Global Face Side

    FToE = zeros(Int64, 2, nfaces)
    FToLF = zeros(Int64, 2, nfaces)
    EToO = Array{Bool, 2}(undef, 4, nelems)
    EToS = zeros(Int64, 4, nelems)

    # Local Face to Local Vertex map
    LFToLV = flatten_tuples(((1, 3), (2, 4), (1, 2), (3, 4)))
    for e in 1:nelems
        for lf in 1:4
            gf = EToF[lf, e]
            if FToE[1, gf] == 0
                @assert FToLF[1, gf] == 0
                FToE[1, gf] = e
                FToLF[1, gf] = lf
                EToO[lf, e] = true
                EToS[lf, e] = 1
            else
                @assert FToE[2, gf] == 0
                @assert FToLF[2, gf] == 0
                FToE[2, gf] = e
                FToLF[2, gf] = lf
                EToS[lf, e] = 2

                ne = FToE[1, gf]
                nf = FToLF[1, gf]

                nv = EToV[LFToLV[:, nf], ne]
                lv = EToV[LFToLV[:, lf], e]
                if nv == lv
                    EToO[lf, e] = true
                elseif nv[end:-1:1] == lv
                    EToO[lf, e] = false
                else
                    error("problem with connectivity")
                end
            end
        end
    end
    (FToE, FToLF, EToO, EToS)
end

function create_metrics(
    pm,
    Nr,
    Ns,
    xf = (r, s) -> (r, ones(size(r)), zeros(size(r))),
    yf = (r, s) -> (s, zeros(size(s)), ones(size(s))),
)
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp

    # Derivative operators for the metric terms
    @assert pm <= 8
    pp = pm == 6 ? 8 : pm

    r = range(-1, stop = 1, length = Nrp)
    s = range(-1, stop = 1, length = Nsp)

    # Create the mesh
    r = ones(1, Nsp) ⊗ r
    s = s' ⊗ ones(Nrp)
    (x, xr, xs) = xf(r, s)
    (y, yr, ys) = yf(r, s)

    J = xr .* ys - xs .* yr
    @assert minimum(J) > 0

    JI = 1 ./ J

    rx = ys ./ J
    sx = -yr ./ J
    ry = -xs ./ J
    sy = xr ./ J

    hr = mapreduce(hypot, min, xr, yr) / (2Nr)
    hs = mapreduce(hypot, min, xs, ys) / (2Ns)
    hmin = min(hr, hs)

    # variable coefficient matrix components (BAE: these would only be so for
    # isotropic problem?)
    crr = J .* (rx .* rx + ry .* ry)
    crs = J .* (sx .* rx + sy .* ry)
    css = J .* (sx .* sx + sy .* sy)

    #
    # Block surface matrices
    #
    (xf1, yf1) = (view(x, 1, :), view(y, 1, :))
    nx1 = -ys[1, :]
    ny1 = xs[1, :]
    sJ1 = hypot.(nx1, ny1)
    nx1 = nx1 ./ sJ1
    ny1 = ny1 ./ sJ1

    (xf2, yf2) = (view(x, Nrp, :), view(y, Nrp, :))
    nx2 = ys[end, :]
    ny2 = -xs[end, :]
    sJ2 = hypot.(nx2, ny2)
    nx2 = nx2 ./ sJ2
    ny2 = ny2 ./ sJ2

    (xf3, yf3) = (view(x, :, 1), view(y, :, 1))
    nx3 = yr[:, 1]
    ny3 = -xr[:, 1]
    sJ3 = hypot.(nx3, ny3)
    nx3 = nx3 ./ sJ3
    ny3 = ny3 ./ sJ3

    (xf4, yf4) = (view(x, :, Nsp), view(y, :, Nsp))
    nx4 = -yr[:, end]
    ny4 = xr[:, end]
    sJ4 = hypot.(nx4, ny4)
    nx4 = nx4 ./ sJ4
    ny4 = ny4 ./ sJ4

    (
        coord = (x, y),
        facecoord = ((xf1, xf2, xf3, xf4), (yf1, yf2, yf3, yf4)),
        crr = crr,
        css = css,
        crs = crs,
        J = J,
        JI = JI,
        sJ = (sJ1, sJ2, sJ3, sJ4),
        nx = (nx1, nx2, nx3, nx4),
        ny = (ny1, ny2, ny3, ny4),
        rx = rx,
        ry = ry,
        sx = sx,
        sy = sy,
        hmin = hmin,
    )
end

function locoperator(
    p,
    Nr,
    Ns,
    metrics = create_metrics(p, Nr, Ns);
    crr = metrics.crr,
    css = metrics.css,
    crs = metrics.crs,
)
    csr = crs

    J = metrics.J
    JI = metrics.JI

    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp

    # Derivative operators for the rest of the computation
    (Dr, HrI, Hr, r) = DiagonalSBP.D1(p, Nr; xc = (-1, 1))
    Qr = Hr * Dr
    QrT = sparse(transpose(Qr))
    Dr = sparse(Dr)
    DrT = sparse(transpose(Dr))
    Hr = sparse(Hr)

    (Ds, HsI, Hs, s) = DiagonalSBP.D1(p, Ns; xc = (-1, 1))
    Qs = Hs * Ds
    QsT = sparse(transpose(Qs))
    Ds = sparse(Ds)
    DsT = sparse(transpose(Ds))
    Hs = sparse(Hs)

    # Identity matrices for the computation
    Ir = sparse(I, Nrp, Nrp)
    Is = sparse(I, Nsp, Nsp)

    # Set up the rr derivative matrix
    ISr0 = Array{Int64, 1}(undef, 0)
    JSr0 = Array{Int64, 1}(undef, 0)
    VSr0 = Array{Float64, 1}(undef, 0)
    ISrN = Array{Int64, 1}(undef, 0)
    JSrN = Array{Int64, 1}(undef, 0)
    VSrN = Array{Float64, 1}(undef, 0)

    Ae = DiagonalSBP.variable_D2(p, Nr, rand(Nrp)).A
    S0e = DiagonalSBP.variable_D2(p, Nr, ones(Nrp)).S0
    SNe = DiagonalSBP.variable_D2(p, Nr, ones(Nrp)).SN

    S0eT = sparse(transpose(S0e))
    SNeT = sparse(transpose(SNe))

    IArr = Array{Int64, 1}(undef, Nsp * length(Ae.nzval))
    JArr = Array{Int64, 1}(undef, Nsp * length(Ae.nzval))
    VArr = Array{Float64, 1}(undef, Nsp * length(Ae.nzval))
    stArr = 0

    for j in 1:Nsp
        rng = (j - 1) * Nrp .+ (1:Nrp)

        Ae = DiagonalSBP.variable_D2(p, Nr, crr[rng]).A
        (Ie, Je, Ve) = findnz(Ae)

        IArr[stArr .+ (1:length(Ve))] = Ie .+ (j - 1) * Nrp
        JArr[stArr .+ (1:length(Ve))] = Je .+ (j - 1) * Nrp
        VArr[stArr .+ (1:length(Ve))] = Hs[j, j] * Ve
        stArr += length(Ve)
    end

    Ãrr = sparse(IArr[1:stArr], JArr[1:stArr], VArr[1:stArr], Np, Np)

    # Set up the ss derivative matrix
    Ae = DiagonalSBP.variable_D2(p, Ns, rand(Nsp)).A

    IAss = Array{Int64, 1}(undef, Nrp * length(Ae.nzval))
    JAss = Array{Int64, 1}(undef, Nrp * length(Ae.nzval))
    VAss = Array{Float64, 1}(undef, Nrp * length(Ae.nzval))
    stAss = 0

    for i in 1:Nrp
        rng = i .+ Nrp * (0:Ns)
        Ae = DiagonalSBP.variable_D2(p, Ns, css[rng]).A
        R = Ae - Dr' * Hr * Diagonal(css[rng]) * Dr

        (Ie, Je, Ve) = findnz(Ae)
        IAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
        JAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
        VAss[stAss .+ (1:length(Ve))] = Hr[i, i] * Ve
        stAss += length(Ve)
    end
    Ãss = sparse(IAss[1:stAss], JAss[1:stAss], VAss[1:stAss], Np, Np)

    # Set up the sr and rs derivative matrices
    Ãsr =
        (QsT ⊗ Ir) *
        sparse(1:length(crs), 1:length(crs), view(crs, :)) *
        (Is ⊗ Qr)
    Ãrs =
        (Is ⊗ QrT) *
        sparse(1:length(csr), 1:length(csr), view(csr, :)) *
        (Qs ⊗ Ir)

    Ã = Ãrr + Ãss + Ãrs + Ãsr

    #
    # Boundary point matrices
    #
    Er0 = sparse([1], [1], [1], Nrp, Nrp)
    ErN = sparse([Nrp], [Nrp], [1], Nrp, Nrp)
    Es0 = sparse([1], [1], [1], Nsp, Nsp)
    EsN = sparse([Nsp], [Nsp], [1], Nsp, Nsp)

    er0 = sparse([1], [1], [1], Nrp, 1)
    erN = sparse([Nrp], [1], [1], Nrp, 1)
    es0 = sparse([1], [1], [1], Nsp, 1)
    esN = sparse([Nsp], [1], [1], Nsp, 1)

    er0T = sparse([1], [1], [1], 1, Nrp)
    erNT = sparse([1], [Nrp], [1], 1, Nrp)
    es0T = sparse([1], [1], [1], 1, Nsp)
    esNT = sparse([1], [Nsp], [1], 1, Nsp)

    #
    # Store coefficient matrices on faces as matrices
    #TODO: switching 2 and 3 below??
    crr1 = sparse(Diagonal(crr[1 .+ Nrp * (0:Ns)]))
    crr3 = sparse(Diagonal(crr[1:Nrp]))
    crr2 = sparse(Diagonal(crr[Nrp .+ Nrp * (0:Ns)]))
    crr4 = sparse(Diagonal(crr[Nrp * Ns .+ (1:Nrp)]))

    crs1 = sparse(Diagonal(crs[1 .+ Nrp * (0:Ns)]))
    crs3 = sparse(Diagonal(crs[1:Nrp]))
    crs2 = sparse(Diagonal(crs[Nrp .+ Nrp * (0:Ns)]))
    crs4 = sparse(Diagonal(crs[Nrp * Ns .+ (1:Nrp)]))

    csr1 = sparse(Diagonal(csr[1 .+ Nrp * (0:Ns)]))
    csr3 = sparse(Diagonal(csr[1:Nrp]))
    csr2 = sparse(Diagonal(csr[Nrp .+ Nrp * (0:Ns)]))
    csr4 = sparse(Diagonal(csr[Nrp * Ns .+ (1:Nrp)]))

    css1 = sparse(Diagonal(css[1 .+ Nrp * (0:Ns)]))
    css3 = sparse(Diagonal(css[1:Nrp]))
    css2 = sparse(Diagonal(css[Nrp .+ Nrp * (0:Ns)]))
    css4 = sparse(Diagonal(css[Nrp * Ns .+ (1:Nrp)]))

    # Restriction operators: restrict the volume solution to the face
    L1 = Is ⊗ er0T
    L2 = Is ⊗ erNT
    L3 = es0T ⊗ Ir
    L4 = esNT ⊗ Ir

    #
    # Surface mass matrices
    #
    H1 = Hs
    H1I = HsI

    H2 = Hs
    H2I = HsI

    H3 = Hr
    H3I = HrI

    H4 = Hr
    H4I = HrI

    #
    # Penalty terms: α = θ_H
    #
    (l, θ_H, _, θ_R) = DiagonalSBP.D2_remainder_parameters(p)

    # check below
    #(θ_1, θ_2, θ_3, θ_4) = (H1[1,1], H2[Nsp, Nsp], H3[1,1], H4[Nrp, Nrp])
    #(ζ_1, ζ_2, ζ_3, ζ_4) = (β*H1[1,1], β*H2[Nsp, Nsp], β*H3[1,1], β*H4[Nrp, Nrp])

    #ψmin = reshape((crr + css - sqrt.((crr - css).^2 + 4crs.^2)) / 2, Nrp, Nsp)
    ψmin_r = reshape(crr, Nrp, Nsp)
    ψmin_s = reshape(css, Nrp, Nsp)
    @assert minimum(ψmin_r) > 0
    @assert minimum(ψmin_s) > 0

    hr = 2 / Nr
    hs = 2 / Ns

    ψ1 = ψmin_r[1, :]
    ψ2 = ψmin_r[Nrp, :]
    ψ3 = ψmin_s[:, 1]
    ψ4 = ψmin_s[:, Nsp]
    for k in 2:l
        ψ1 = min.(ψ1, ψmin_r[k, :])
        ψ2 = min.(ψ2, ψmin_r[Nrp + 1 - k, :])
        ψ3 = min.(ψ3, ψmin_s[:, k])
        ψ4 = min.(ψ4, ψmin_s[:, Nsp + 1 - k])
    end

    τR1 = (1 / (θ_R * hr)) * Is
    τR2 = (1 / (θ_R * hr)) * Is
    τR3 = (1 / (θ_R * hs)) * Ir
    τR4 = (1 / (θ_R * hs)) * Ir

    p1 = ((crr[1, :]) ./ ψ1)
    p2 = ((crr[Nrp, :]) ./ ψ2)
    p3 = ((css[:, 1]) ./ ψ3)
    p4 = ((css[:, Nsp]) ./ ψ4)

    P1 = sparse(1:Nsp, 1:Nsp, p1)
    P2 = sparse(1:Nsp, 1:Nsp, p2)
    P3 = sparse(1:Nrp, 1:Nrp, p3)
    P4 = sparse(1:Nrp, 1:Nrp, p4)
    # Penalty matrices:
    Γ1 = 1 * ((2 / (θ_H * hr)) * Is + τR1 * P1)
    Γ2 = 1 * ((2 / (θ_H * hr)) * Is + τR2 * P2)
    Γ3 = 1 * ((2 / (θ_H * hs)) * Ir + τR3 * P3)
    Γ4 = 1 * ((2 / (θ_H * hs)) * Ir + τR4 * P4)

    # Boundary matrices on faces
    Brr_1 = crr1 * L1 * (Is ⊗ S0e)
    Brr_2 = crr2 * L2 * (Is ⊗ SNe)
    Brr_3 = crr3 * L3 * (Is ⊗ Dr)
    Brr_4 = crr4 * L4 * (Is ⊗ Dr)
    Brr_1T = (Is ⊗ S0eT) * L1' * crr1
    Brr_2T = (Is ⊗ SNeT) * L2' * crr2
    Brr_3T = (Is ⊗ DrT) * L3' * crr3
    Brr_4T = (Is ⊗ DrT) * L4' * crr4

    Brs_1 = crs1 * L1 * (Ds ⊗ Ir) #TODO: START FIXING  HERE
    Brs_2 = crs2 * L2 * (Ds ⊗ Ir)
    Brs_3 = crs3 * L3 * (S0e ⊗ Ir)
    Brs_4 = crs4 * L4 * (SNe ⊗ Ir)
    Brs_1T = (DsT ⊗ Ir) * L1' * crs1
    Brs_2T = (DsT ⊗ Ir) * L2' * crs2
    Brs_3T = (S0eT ⊗ Ir) * L3' * crs3
    Brs_4T = (SNeT ⊗ Ir) * L4' * crs4

    Bsr_1 = csr1 * L1 * (Is ⊗ S0e)
    Bsr_2 = csr2 * L2 * (Is ⊗ SNe)
    Bsr_3 = csr3 * L3 * (Is ⊗ Dr)
    Bsr_4 = csr4 * L4 * (Is ⊗ Dr)
    Bsr_1T = (Is ⊗ S0eT) * L1' * csr1
    Bsr_2T = (Is ⊗ SNeT) * L2' * csr2
    Bsr_3T = (Is ⊗ DrT) * L3' * csr3
    Bsr_4T = (Is ⊗ DrT) * L4' * csr4

    Bss_1 = css1 * L1 * (Ds ⊗ Ir)
    Bss_2 = css2 * L2 * (Ds ⊗ Ir)
    Bss_3 = css3 * L3 * (S0e ⊗ Ir)
    Bss_4 = css4 * L4 * (SNe ⊗ Ir)
    Bss_1T = (DsT ⊗ Ir) * L1' * css1
    Bss_2T = (DsT ⊗ Ir) * L2' * css2
    Bss_3T = (S0eT ⊗ Ir) * L3' * css3
    Bss_4T = (SNeT ⊗ Ir) * L4' * css4

    HfI = (H1I, H2I, H3I, H4I)

    JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)
    JHI = sparse(1:Np, 1:Np, view(J, :)) * (HsI ⊗ HrI)
    J = sparse(1:Np, 1:Np, view(J, :))
    JI = sparse(1:Np, 1:Np, view(JI, :))
    HI = (HsI ⊗ HrI)
    H = (Hs ⊗ Hr)

    (
        Ã = Ã,
        coord = metrics.coord,
        facecoord = metrics.facecoord,
        JH = JH,
        JHI = JHI,
        sJ = metrics.sJ,
        nx = metrics.nx,
        ny = metrics.ny,
        Hf = (H1, H2, H3, H4),
        L = (L1, L2, L3, L4),
        Crrf = (crr1, crr2, crr3, crr4),
        Crsf = (crs1, crs2, crs3, crs4),
        Csrf = (csr1, csr2, csr3, csr4),
        Cssf = (css1, css2, css3, css4),
        Brr = (Brr_1, Brr_2, Brr_3, Brr_4, Brr_1T, Brr_2T, Brr_3T, Brr_4T),
        Brs = (Brs_1, Brs_2, Brs_3, Brs_4, Brs_1T, Brs_2T, Brs_3T, Brs_4T),
        Bsr = (Bsr_1, Bsr_2, Bsr_3, Bsr_4, Bsr_1T, Bsr_2T, Bsr_3T, Bsr_4T),
        Bss = (Bss_1, Bss_2, Bss_3, Bss_4, Bss_1T, Bss_2T, Bss_3T, Bss_4T),
        Γ = (Γ1, Γ2, Γ3, Γ4),
        HI = HI,
        H = H,
        J = J,
        JI = JI,
        HfI = (H1I, H2I, H3I, H4I),
    )
end

function rhsoperators(
    p,
    Nqr,
    Nqs,
    metrics,
    gDfun,
    gDdotfun,
    gNfun,
    F = nothing;
    rho = 1,
    crr = metrics.crr,
    css = metrics.css,
    crs = metrics.crs,
)

    # Build local operators

    lop = locoperator(
        p,
        Nqr - 1,
        Nqs - 1,
        metrics;
        crr = crr,
        css = css,
        crs = crs,
    )

    Np = Nqr * Nqs
    Ã = lop.Ã

    # Surface mass matrices
    H = lop.Hf

    (xf1, xf2, xf3, xf4) = metrics.facecoord[1]
    (yf1, yf2, yf3, yf4) = metrics.facecoord[2]

    #Surface normal vectors
    nx = metrics.nx
    ny = metrics.ny #DON'T assume normal vectors don't vary on face

    #hat normal vectors on 4 faces:
    nr = (-1, 1, 0, 0)
    ns = (0, 0, -1, 1)

    sJ = metrics.sJ

    # coefficients on the 4 faces
    crr = lop.Crrf
    crs = lop.Crsf
    csr = lop.Csrf
    css = lop.Cssf

    Z1 = sqrt.(rho .* (nx[1] .* nx[1] .+ ny[1] .* ny[1]))
    Z2 = sqrt.(rho .* (nx[2] .* nx[2] .+ ny[2] .* ny[2]))
    Z3 = sqrt.(rho .* (nx[3] .* nx[3] .+ ny[3] .* ny[3]))
    Z4 = sqrt.(rho .* (nx[4] .* nx[4] .+ ny[4] .* ny[4]))
    Z = (Z1, Z2, Z3, Z4)

    Ẑ1 = sJ[1] .* Z1
    Ẑ2 = sJ[2] .* Z2
    Ẑ3 = sJ[3] .* Z3
    Ẑ4 = sJ[4] .* Z4

    #Ẑ = (Ẑ1, Ẑ2, Ẑ3, Ẑ4)
    Ẑ = (
        spdiagm(0 => Ẑ1),
        spdiagm(0 => Ẑ2),
        spdiagm(0 => Ẑ3),
        spdiagm(0 => Ẑ4),
    )

    Ẑinv = (
        spdiagm(0 => 1 ./ Ẑ1),
        spdiagm(0 => 1 ./ Ẑ2),
        spdiagm(0 => 1 ./ Ẑ3),
        spdiagm(0 => 1 ./ Ẑ4),
    )

    # interpolation operators
    L = lop.L
    Γ = lop.Γ  # Need to fix these to be correct

    HI = lop.HI
    JH = lop.JH
    H̃ = lop.H
    JHI = lop.JHI
    H = lop.Hf
    J = lop.J
    JI = lop.JI
    JIHI = JI * HI

    # boundary deriv ops - include coefficients
    Brr = lop.Brr[1:4]
    BrrT = lop.Brr[5:8]
    Brs = lop.Brs[1:4]
    BrsT = lop.Brs[5:8]
    Bsr = lop.Bsr[1:4]
    BsrT = lop.Bsr[5:8]
    Bss = lop.Bss[1:4]
    BssT = lop.Bss[5:8]

    nCB = (
        nr[1] * (Brr[1] + Brs[1]) + ns[1] * (Bsr[1] + Bss[1]),
        nr[2] * (Brr[2] + Brs[2]) + ns[2] * (Bsr[2] + Bss[2]),
        nr[3] * (Brr[3] + Brs[3]) + ns[3] * (Bsr[3] + Bss[3]),
        nr[4] * (Brr[4] + Brs[4]) + ns[4] * (Bsr[4] + Bss[4]),
    )

    nCnΓ = (
        (
            nr[1] * crr[1] * nr[1] +
            nr[1] * crs[1] * ns[1] +
            ns[1] * csr[1] * nr[1] +
            ns[1] * css[1] * ns[1]
        ) * Γ[1],
        (
            nr[2] * crr[2] * nr[2] +
            nr[2] * crs[2] * ns[2] +
            ns[2] * csr[2] * nr[2] +
            ns[2] * css[2] * ns[2]
        ) * Γ[2],
        (
            nr[3] * crr[3] * nr[3] +
            nr[3] * crs[3] * ns[3] +
            ns[3] * csr[3] * nr[3] +
            ns[3] * css[3] * ns[3]
        ) * Γ[3],
        (
            nr[4] * crr[4] * nr[4] +
            nr[4] * crs[4] * ns[4] +
            ns[4] * csr[4] * nr[4] +
            ns[4] * css[4] * ns[4]
        ) * Γ[4],
    )

    nCnΓL = (
        (
            nr[1] * crr[1] * nr[1] +
            nr[1] * crs[1] * ns[1] +
            ns[1] * csr[1] * nr[1] +
            ns[1] * css[1] * ns[1]
        ) *
        Γ[1] *
        L[1],
        (
            nr[2] * crr[2] * nr[2] +
            nr[2] * crs[2] * ns[2] +
            ns[2] * csr[2] * nr[2] +
            ns[2] * css[2] * ns[2]
        ) *
        Γ[2] *
        L[2],
        (
            nr[3] * crr[3] * nr[3] +
            nr[3] * crs[3] * ns[3] +
            ns[3] * csr[3] * nr[3] +
            ns[3] * css[3] * ns[3]
        ) *
        Γ[3] *
        L[3],
        (
            nr[4] * crr[4] * nr[4] +
            nr[4] * crs[4] * ns[4] +
            ns[4] * csr[4] * nr[4] +
            ns[4] * css[4] * ns[4]
        ) *
        Γ[4] *
        L[4],
    )

    BtnCH = (
        (nr[1] * (BrrT[1] + BrsT[1]) + ns[1] * (BsrT[1] + BssT[1])) * H[1],
        (nr[2] * (BrrT[2] + BrsT[2]) + ns[2] * (BsrT[2] + BssT[2])) * H[2],
        (nr[3] * (BrrT[3] + BrsT[3]) + ns[3] * (BsrT[3] + BssT[3])) * H[3],
        (nr[4] * (BrrT[4] + BrsT[4]) + ns[4] * (BsrT[4] + BssT[4])) * H[4],
    )

    X1 =
        spdiagm(0 => 1 ./ diag(Γ[1])) *
        ((1 / nr[1]) * spdiagm(0 => 1 ./ diag(crr[1])) * (1 / nr[1]))
    X2 =
        spdiagm(0 => 1 ./ diag(Γ[2])) *
        ((1 / nr[2]) * spdiagm(0 => 1 ./ diag(crr[2])) * (1 / nr[2]))
    X3 =
        spdiagm(0 => 1 ./ diag(Γ[3])) *
        ((1 / ns[3]) * spdiagm(0 => 1 ./ diag(css[3])) * (1 / ns[3]))
    X4 =
        spdiagm(0 => 1 ./ diag(Γ[4])) *
        ((1 / ns[4]) * spdiagm(0 => 1 ./ diag(css[4])) * (1 / ns[4]))

    X = (X1, X2, X3, X4)

    ue_n =
        (t, e) -> (
            gNfun.(nx[1], ny[1], xf1[:], yf1[:], t, e),
            gNfun.(nx[2], ny[2], xf2[:], yf2[:], t, e),
            gNfun.(nx[3], ny[3], xf3[:], yf3[:], t, e),
            gNfun.(nx[4], ny[4], xf4[:], yf4[:], t, e),
        )

    gD =
        (t, e) -> (
            gDfun.(xf1[:], yf1[:], t, e),
            gDfun.(xf2[:], yf2[:], t, e),
            gDfun.(xf3[:], yf3[:], t, e),
            gDfun.(xf4[:], yf4[:], t, e),
        )
    gDdot =
        (t, e) -> (
            gDdotfun.(xf1[:], yf1[:], t, e),
            gDdotfun.(xf2[:], yf2[:], t, e),
            gDdotfun.(xf3[:], yf3[:], t, e),
            gDdotfun.(xf4[:], yf4[:], t, e),
        )

    sJgDdot =
        (t, e) -> (
            spdiagm(0 => sJ[1]) * gDdot(t, e)[1],
            spdiagm(0 => sJ[2]) * gDdot(t, e)[2],
            spdiagm(0 => sJ[3]) * gDdot(t, e)[3],
            spdiagm(0 => sJ[4]) * gDdot(t, e)[4],
        )

    gN =
        (t, e) -> (
            spdiagm(0 => sJ[1]) * ue_n(t, e)[1],
            spdiagm(0 => sJ[2]) * ue_n(t, e)[2],
            spdiagm(0 => sJ[3]) * ue_n(t, e)[3],
            spdiagm(0 => sJ[4]) * ue_n(t, e)[4],
        )

    if isnothing(F)
        myF = nothing
    else
        myF = (t, e) -> F.(metrics.coord[1][:], metrics.coord[2][:], t, e)
    end
    (
        Ã = Ã,
        nCB = nCB,
        nCnΓ = nCnΓ,
        nCnΓL = nCnΓL,
        gD = gD,
        gDdot = gDdot,
        Z = Ẑ,
        L = L,
        gN = gN,
        sJgDdot = sJgDdot,
        Zinv = Ẑinv,
        H = H,
        BtnCH = BtnCH,
        JIHI = JIHI,
        F = myF,
        JH = JH,
        sJ = sJ,
        X = X,
        H̃ = H̃,
        J = J,
    )
end

# Constructor for inp files
function read_inp_2d(T, S, filename::String; bc_map = 1:10000)
    # Read in the file
    f = try
        open(filename)
    catch
        error("InpRead cannot open \"$filename\" ")
    end
    lines = readlines(f)
    close(f)

    # Read in nodes
    str = "NSET=ALLNODES"
    linenum = SeekToSubstring(lines, str)
    linenum > 0 || error("did not find: $str")
    num_nodes = 0
    for l in (linenum + 1):length(lines)
        occursin(r"^\s*[0-9]*\s*,.*", lines[l]) ? num_nodes += 1 : break
    end
    Vx = fill(S(NaN), num_nodes)
    Vy = fill(S(NaN), num_nodes)
    Vz = fill(S(NaN), num_nodes)
    for l in linenum .+ (1:num_nodes)
        node_data = split(lines[l], r"\s|,", keepempty = false)
        (node_num, node_x, node_y, node_z) = try
            (
                parse(T, node_data[1]),
                parse(S, node_data[2]),
                parse(S, node_data[3]),
                parse(S, node_data[4]),
            )
        catch
            error("cannot parse line $l: \"$(lines[l])\" ")
        end

        Vx[node_num] = node_x
        Vy[node_num] = node_y
        Vz[node_num] = node_z
    end

    # Read in Elements
    str = "ELEMENT"
    linenum = SeekToSubstring(lines, str)
    num_elm = 0
    while linenum > 0
        for l in linenum .+ (1:length(lines))
            occursin(r"^\s*[0-9]*\s*,.*", lines[l]) ? num_elm += 1 : break
        end
        linenum = SeekToSubstring(lines, str; first = linenum + 1)
    end
    num_elm > 0 || error("did not find any element")

    EToV = fill(T(0), 4, num_elm)
    EToBlock = fill(T(0), num_elm)
    linenum = SeekToSubstring(lines, str)
    while linenum > 0
        foo = split(lines[linenum], r"[^0-9]", keepempty = false)
        B = parse(T, foo[end])
        for l in linenum .+ (1:num_elm)
            elm_data = split(lines[l], r"\s|,", keepempty = false)
            # read into z-order
            (elm_num, elm_v1, elm_v2, elm_v4, elm_v3) = try
                (
                    parse(T, elm_data[1]),
                    parse(T, elm_data[2]),
                    parse(T, elm_data[3]),
                    parse(T, elm_data[4]),
                    parse(T, elm_data[5]),
                )
            catch
                break
            end
            EToV[:, elm_num] = [elm_v1, elm_v2, elm_v3, elm_v4]
            EToBlock[elm_num] = B
        end
        linenum = SeekToSubstring(lines, str; first = linenum + 1)
    end

    # Determine connectivity
    EToF = fill(T(0), 4, num_elm)

    VsToF = Dict{Tuple{Int64, Int64}, Int64}()
    numfaces = 0
    for e in 1:num_elm
        for lf in 1:4
            if lf == 1
                Vs = (EToV[1, e], EToV[3, e])
            elseif lf == 2
                Vs = (EToV[2, e], EToV[4, e])
            elseif lf == 3
                Vs = (EToV[1, e], EToV[2, e])
            elseif lf == 4
                Vs = (EToV[3, e], EToV[4, e])
            end
            if Vs[1] > Vs[2]
                Vs = (Vs[2], Vs[1])
            end
            if haskey(VsToF, Vs)
                EToF[lf, e] = VsToF[Vs]
            else
                numfaces = numfaces + 1
                EToF[lf, e] = VsToF[Vs] = numfaces
            end
        end
    end

    # Read in side set info
    FToB = Array{T, 1}(undef, numfaces)
    fill!(FToB, BC_LOCKED_INTERFACE)
    linenum = SeekToSubstring(lines, "\\*ELSET")
    inp_to_zorder = [3, 2, 4, 1]
    while linenum > 0
        foo = split(lines[linenum], r"[^0-9]", keepempty = false)
        (bc, face) = try
            (parse(T, foo[1]), parse(T, foo[2]))
        catch
            error("cannot parse line $linenum: \"$(lines[linenum])\" ")
        end
        bc = bc_map[bc]
        face = inp_to_zorder[face]
        for l in (linenum + 1):length(lines)
            if !occursin(r"^\s*[0-9]+", lines[l])
                break
            end
            elms = split(lines[l], r"\s|,", keepempty = false)
            for elm in elms
                elm = try
                    parse(T, elm)
                catch
                    error("cannot parse line $linenum: \"$(lines[l])\" ")
                end
                if bc == 3
                    bc = BC_LOCKED_INTERFACE
                end
                FToB[EToF[face, elm]] = bc
                @assert (
                    bc == BC_DIRICHLET ||
                    bc == BC_NEUMANN ||
                    bc == BC_LOCKED_INTERFACE ||
                    abs(bc) >= BC_JUMP_INTERFACE
                )
            end
        end
        linenum = SeekToSubstring(lines, "\\*ELSET"; first = linenum + 1)
    end

    ([Vx Vy]', EToV, EToF, FToB, EToBlock)
end
read_inp_2d(filename; kw...) = read_inp_2d(Int64, Float64, filename; kw...)

function SeekToSubstring(lines, substring; first = 1)
    for l in first:length(lines)
        if occursin(Regex(".*$(substring).*"), lines[l])
            return l
        end
    end
    return -1
end

function plot_connectivity(verts, EToV)
    Lx = extrema(verts[1, :])
    Lx = (floor(Int, Lx[1]), ceil(Int, Lx[2]))
    Ly = extrema(verts[2, :])
    Ly = (floor(Int, Ly[1]), ceil(Int, Ly[2]))
    width = Lx[2] - Lx[1]
    height = Ly[2] - Ly[1]
    plt = Plot(
        BrailleCanvas(
            80,
            ceil(Int, 40 * height / width),
            origin_x = Lx[1],
            origin_y = Ly[1],
            width = width,
            height = height,
        ),
    )

    annotate!(plt, :l, nrows(plt.graphics), string(Ly[1]), color = :light_black)
    annotate!(plt, :l, 1, string(Ly[2]), color = :light_black)
    annotate!(plt, :bl, string(Lx[1]), color = :light_black)
    annotate!(plt, :br, string(Lx[2]), color = :light_black)
    for e in 1:size(EToV, 2)
        (v1, v2, v3, v4) = EToV[1:4, e]
        x = verts[1, [v1 v2 v4 v3 v1]][:]
        y = verts[2, [v1 v2 v4 v3 v1]][:]
        lineplot!(plt, x, y)
    end
    title!(plt, "connectivity")
    display(plt)
end

function plot_blocks(lop, FToB, EToF)
    Lx = (floatmax(), -floatmax())
    Ly = (floatmax(), -floatmax())
    for e in 1:length(lop)
        (x, y) = lop[e].coord
        Lxe = extrema(x)
        Lye = extrema(y)
        Lx = (min(Lx[1], Lxe[1]), max(Lx[2], Lxe[2]))
        Ly = (min(Ly[1], Lye[1]), max(Ly[2], Lye[2]))
    end

    Lx = (floor(Int, Lx[1]), ceil(Int, Lx[2]))
    Ly = (floor(Int, Ly[1]), ceil(Int, Ly[2]))

    width = Lx[2] - Lx[1]
    height = Ly[2] - Ly[1]
    plt = Plot(
        BrailleCanvas(
            80,
            ceil(Int, 40 * height / width),
            origin_x = Lx[1],
            origin_y = Ly[1],
            width = width,
            height = height,
        ),
    )

    annotate!(plt, :l, nrows(plt.graphics), string(Ly[1]), color = :light_black)
    annotate!(plt, :l, 1, string(Ly[2]), color = :light_black)
    annotate!(plt, :bl, string(Lx[1]), color = :light_black)
    annotate!(plt, :br, string(Lx[2]), color = :light_black)

    for e in 1:length(lop)
        (xf, yf) = lop[e].facecoord
        for lf in 1:length(xf)
            bc_type_face = FToB[EToF[lf, e]]
            if bc_type_face == BC_LOCKED_INTERFACE
                lineplot!(plt, xf[lf], yf[lf], color = :blue)
            elseif bc_type_face == BC_DIRICHLET
                lineplot!(plt, xf[lf], yf[lf], color = :green)
            elseif bc_type_face == BC_DIRICHLET
                lineplot!(plt, xf[lf], yf[lf], color = :yellow)
            else
                lineplot!(plt, xf[lf], yf[lf], color = :red)
            end
        end
    end
    title!(plt, "mesh")
    display(plt)
end

function write_vtk(
    output_file,
    metrics,
    q;
    cxx = nothing,
    cyy = nothing,
    cxy = nothing,
)
    vtk_multiblock(output_file) do vtm
        # How many blocks do we have
        nblocks = length(metrics)

        # loop over the blocks and output for this block

        # pointer for end of the last block
        n = 0
        for block in 1:nblocks
            # get the grid
            x = metrics[block].coord[1]
            y = metrics[block].coord[2]

            # get the number of points associated with this block
            Nr, Ns = size(x)
            Np = Nr * Ns

            # get the length of the q vector for this block
            q_len = 2(Np + Nr + Ns)

            # get the displacements

            vtk = vtk_grid(vtm, x, y)
            vtk["u"] = @view q[n .+ (1:Np)]
            vtk["v"] = @view q[(n + Np) .+ (1:Np)]

            if !isnothing(cxx)
                vtk["cxx"] = cxx.(x, y)
                vtk["cxy"] = cxy.(x, y)
                vtk["cyy"] = cyy.(x, y)
            end

            # update the last point of the block
            n += q_len
        end

        # We should have hit all the data by now
        @assert n == length(q)
    end
end

# This uses the fourth-order, low-storage, Runge--Kutta scheme of
# Carpenter and Kennedy (1994) (in their notation (5,4) 2N-Storage RK
#                               scheme).
#
# @TECHREPORT{CarpenterKennedy1994,
#             author = {M.~H. Carpenter and C.~A. Kennedy},
#             title = {Fourth-order {2N-storage} {Runge-Kutta}
#                      schemes},
#             institution = {National Aeronautics and Space
#                            Administration},
#             year = {1994},
#             number = {NASA TM-109112},
#             address = {Langley Research
#                        Center, Hampton, VA},
#            }
function timestep!(q, f!, p, dt, (t0, t1), Δq = similar(q), Δq2 = similar(q))
    T = eltype(q)

    RKA = (
        T(0),
        T(-567301805773 // 1357537059087),
        T(-2404267990393 // 2016746695238),
        T(-3550918686646 // 2091501179385),
        T(-1275806237668 // 842570457699),
    )

    RKB = (
        T(1432997174477 // 9575080441755),
        T(5161836677717 // 13612068292357),
        T(1720146321549 // 2090206949498),
        T(3134564353537 // 4481467310338),
        T(2277821191437 // 14882151754819),
    )

    RKC = (
        T(0),
        T(1432997174477 // 9575080441755),
        T(2526269341429 // 6820363962896),
        T(2006345519317 // 3224310063776),
        T(2802321613138 // 2924317926251),
    )

    nstep = ceil(Int, (t1 - t0) / dt)
    dt = (t1 - t0) / nstep

    fill!(Δq, 0)
    fill!(Δq2, 0)
    for step in 1:nstep
        t = t0 + (step - 1) * dt
        for s in 1:length(RKA)
            f!(Δq2, q, p, t + RKC[s] * dt)
            Δq .+= Δq2
            q .+= RKB[s] * dt * Δq
            Δq .*= RKA[s % length(RKA) + 1]
        end
    end

    nothing
end

# Based on the zero algorithm of  Richard P.  Brent, Algorithms for Minimization
# without Derivatives, Prentice-Hall, Englewood Cliffs, New Jersey, 1973, 195
# pp. (available online at
# https://maths-people.anu.edu.au/~brent/pub/pub011.html)
function brentdekker(
    f,
    a::T,
    b::T,
    xatol::T = eps(T),
    max_iterations = 100,
    xrtol::T = xatol,
    ftol::T = T(0),
) where {T <: AbstractFloat}

    # Evaluate the function 
    fa, fb = f(a), f(b)

    # Not a bracket
    if fa * fb > 0
        return (b, false, -1)
    end

    # Initialize the other side of the bracket
    c, fc = a, fa

    # below d is the current step and e is previous step
    d = e = b - a

    macheps = eps(T)

    # solution is in bracket [c, b] and a is the previous b
    for i in 1:max_iterations

        # `b` should be best guess (even if that means we don't update value)
        # This also resets `a` (the previous `b`) to be the same as `c
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end

        # tol is used to make sure that we are properly scaled with respect to
        # the input
        tol = 2xrtol * abs(b) + xatol

        # bisection step size
        m = (c - b) / 2

        # If the bracket is small enough or we hit the root
        if (abs(m) < tol || fb == 0 || abs(fb) < ftol)
            return (b, true, i)
        end

        # if the step we took last time `e` is smaller than the tolerance OR if
        # our function really increased just do bisection
        if abs(e) < tol || abs(fa) < abs(fb)
            d = e = m
        else
            # do inverse quadratic interpolation if we can
            # otherwise do secant
            # The use of p & q is to make things more floating point stable

            r = fb / fc
            if a ≠ c
                s = fb / fa
                q = fa / fc
                p = s * (2m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            else
                p = 2m * r
                q = 1 - r
            end

            # either p or q has to flip signs, since we want p to be positive
            # below we prefer to flip the sign of q
            p > 0 ? (q = -q) : (p = -p)

            # As long as the step keeps us in the bracket and the step is not too
            # large we accept the secant / quadratic step
            #
            # The first condition ensures that the step p / q cause the solution
            # to be in the interval [b, b + (3/4)*(a-b)], e.g, we do not get too
            # close to a which is the worst of the two sides.
            #
            # The second condition ensures that the step is at least half as big
            # as the previous step, e.g., p / q < e / 2 (otherwise we are not
            # converging quickly so we should just bisect)
            if 2p < 3m * q - abs(tol * q) && p < abs(e * q / 2)
                e, d = d, p / q
            else
                d = e = m
            end
        end

        # Save the last step
        a, fa = b, fb

        # As long as the step isn't too small accept it, otherwise take a `tol`
        # sizes step in the correct direction
        b = b + (abs(d) > tol ? d : (m > 0 ? tol : -tol))
        fb = f(b)

        # if fb and fc have same sign then really a, b bracket the root
        if (fb > 0) == (fc > 0)
            c, fc = a, fa
            d = e = b - a
        end
    end

    return (b, false, max_iterations)
end

function waveprop!(dq, q, params, t)
    Nqr = params.Nqr
    Nqs = params.Nqs
    rhsops = params.rhsops
    EToF = params.EToF
    EToO = params.EToO
    EToS = params.EToS
    FToB = params.FToB
    FToE = params.FToE
    FToLF = params.FToLF
    friction = params.friction

    nelems = length(rhsops)
    Np = Nqr * Nqs

    lenq0 = 2 * (Np + Nqr + Nqs)

    # @show t

    @assert length(q) == nelems * lenq0
    @inbounds Threads.@threads for e in 1:nelems
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

        qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]
        dqe = @view dq[(lenq0 * (e - 1) + 1):(e * lenq0)]

        u = @view qe[1:Np]
        v = @view qe[Np .+ (1:Np)]
        û1 = @view qe[(2Np + 1):(2Np + Nqs)]
        û2 = @view qe[(2Np + 1 + Nqs):(2(Np + Nqs))]
        û3 = @view qe[(2(Np + Nqs) + 1):(2(Np + Nqs) + Nqr)]
        û4 = @view qe[(2(Np + Nqs) + Nqr + 1):(2(Np + Nqs + Nqr))]

        ustar = (û1, û2, û3, û4)

        du = @view dqe[1:Np]
        dv = @view dqe[Np .+ (1:Np)]
        dû1 = @view dqe[(2Np + 1):(2Np + Nqs)]
        dû2 = @view dqe[(2Np + 1 + Nqs):(2(Np + Nqs))]
        dû3 = @view dqe[(2(Np + Nqs) + 1):(2(Np + Nqs) + Nqr)]
        dû4 = @view dqe[(2(Np + Nqs) + Nqr + 1):(2(Np + Nqs + Nqr))]

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

        glb_fcs = EToF[:, e]  #global faces of element e

        for lf in 1:4
            bc_type_face = FToB[glb_fcs[lf]]
            if bc_type_face == BC_DIRICHLET
                τ̂star[lf] .= τ̂[lf]
                vstar[lf] .= rhsops[e].gDdot(t, e)[lf]

            elseif bc_type_face == BC_NEUMANN
                τ̂star[lf] .= rhsops[e].gN(t, e)[lf]
                vstar[lf] .= rhsops[e].L[lf] * v

            elseif bc_type_face == BC_JUMP_INTERFACE
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
                uplus = @view qplus[1:Np]
                vplus = @view qplus[Np .+ (1:Np)]
                û1plus = qplus[(2Np + 1):(2Np + Nqs)]
                û2plus = qplus[(2Np + 1 + Nqs):(2(Np + Nqs))]
                û3plus = qplus[(2(Np + Nqs) + 1):(2(Np + Nqs) + Nqr)]
                û4plus = qplus[(2(Np + Nqs) + Nqr + 1):(2(Np + Nqs + Nqr))]

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
                    rhsops[e].sJ[lf] .* friction.(Vminus)
                sJgplus = gNplus_rev - sJplus_rev .* friction.(-Vminus)

                #gminus = rhsops[e].Zinv[lf] * sJgminus
                #gplus = Z_refI * sJgplus

                Zvminus = Z * rhsops[e].L[lf] * v
                Zvplus = Z_rev * vplus_fc_rev

                vfric = zeros(Float64, Nqr)

                for i in 1:Nqr
                    f(V) =
                        2 * sJminus[i] * friction(V) + Z[i, i] * V -
                        (Zvplus[i] - Zvminus[i]) + τ̂plus_rev[i] - τ̂[lf][i] -
                        sJgplus[i] + sJgminus[i]

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
                        rhsops[e].sJ[lf] .* friction.(vfric) +
                        sJgminus
                    )
                τ̂star[lf] .=
                    rhsops[e].Z[lf] * (vstar[lf] - rhsops[e].L[lf] * v) + τ̂[lf]

            elseif bc_type_face == -BC_JUMP_INTERFACE

                els_share = FToE[:, glb_fcs[lf]]
                lf_share_glb = FToLF[:, glb_fcs[lf]]
                cel = [1]
                cfc = [1]
                for i in 1:2
                    if els_share[i] != e
                        cel[1] = els_share[i] # other element face is connected to!
                    end
                end
                gDdotplus = rhsops[cel[1]].gDdot(t, cel[1])[cfc[1]]
                qplus = @view q[(lenq0 * (cel[1] - 1) + 1):(cel[1] * lenq0)]
                vplus = @view qplus[Np .+ (1:Np)]
                vplus = rhsops[cel[1]].L[cfc[1]] * vplus
                if EToO[lf, e] != EToO[cfc[1], cel[1]]
                  gDdotplus .= gDdotplus[end:-1:1]
                  vplus .= vplus[end:-1:1]
                end
                Vdisc = vplus - rhsops[e].L[lf] * v
                Vexact = gDdotplus - rhsops[e].gDdot(t, e)[lf]
                τ̂star[lf] .=
                rhsops[e].gN(t, e)[lf] - rhsops[e].sJ[lf] .* (friction.(Vexact) - friction.(Vdisc))
                vstar[lf] .= rhsops[e].L[lf] * v

            elseif bc_type_face == BC_LOCKED_INTERFACE
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
                uplus = @view qplus[1:Np]
                vplus = @view qplus[Np .+ (1:Np)]
                û1plus = qplus[(2Np + 1):(2Np + Nqs)]
                û2plus = qplus[(2Np + 1 + Nqs):(2(Np + Nqs))]
                û3plus = qplus[(2(Np + Nqs) + 1):(2(Np + Nqs) + Nqr)]
                û4plus = qplus[(2(Np + Nqs) + Nqr + 1):(2(Np + Nqs + Nqr))]

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
        # dv .= -rhsops[e].Ã * u
        mul!(dv, rhsops[e].Ã, u, -1, 0)

        for lf in 1:4
            dv .+=
                rhsops[e].L[lf]' * (rhsops[e].H[lf] * τ̂star[lf]) -
                rhsops[e].BtnCH[lf] * (ustar[lf] - rhsops[e].L[lf] * u)
        end

        if isnothing(rhsops[e].F)
            dv .= rhsops[e].JIHI * dv
        else
            dv .= rhsops[e].JIHI * dv .+ rhsops[e].F(t, e)
        end

        dû1 .= vstar[1]
        dû2 .= vstar[2]
        dû3 .= vstar[3]
        dû4 .= vstar[4]
    end
end
