using DiagonalSBP
using PGFPlotsX: @pgf, Axis, Plot, pgfsave, Table, LogLogAxis
using ForwardDiff: derivative
using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using SparseArrays: sparsevec
using Printf: @sprintf

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

function bisection(F, a::T, b::T, tol = 100 * eps(T)) where T
    (a, b) = a < b ? (a, b) : (b, a)

    Fa = F(a)
    Fb = F(b)
    if abs(Fa) < abs(Fb)
        abs(Fa) < eps(T) && return a
    else
        abs(Fb) < eps(T) && return b
    end
    @assert Fa * Fb < 0

    V = (a + b) / 2
    while b - a > tol
        FV = F(V)
        if FV == 0
            break
        elseif FV * Fa < 0
            b = V
        else
            a = V
        end
        V = (a + b) / 2
    end

    return V
end

function semianalytic_solution(u0, v0, σ0, tend::T, F) where T

    # Characteristics propagating into interface
    w_m(x, t) = v0(x - t) - σ0(x - t)
    w_p(x, t) = v0(x + t) + σ0(x + t) # zero

    # Find V by root finding
    η(t) = w_p(0, t) - w_m(0, t)
    g(V, t) = 2F(V) + V - η(t)
    V(t) = t > 0 ? bisection(V -> g(V, t), zero(T), η(t)) : zero(t)

    # Set Characteristics propagating out of interface
    Q_m(t) = w_m(0, t) + 2F(V(t))
    Q_p(t) = w_p(0, t) - 2F(V(t))

    # Integrate variables in order to compute displacement
    ∫Q_m_ = solve(ODEProblem((_, _, t) -> Q_m(t), zero(T), (zero(T), tend)),
                  Tsit5(),
                  dtmax = 0.1,
                  abstol=1e-12,
                  reltol=1e-12)
    ∫Q_m(t) = t > 0 ? ∫Q_m_(t) : zero(T)
    ∫Q_p_ = solve(ODEProblem((_, _, t) -> Q_p(t), zero(T), (zero(T), tend)),
                  Tsit5(),
                  dtmax = 0.1,
                  abstol=1e-12,
                  reltol=1e-12)
    ∫Q_p(t) = t > 0 ? ∫Q_p_(t) : zero(T)

    ∫q_m(x, t) = ∫Q_m(t + x)
    ∫q_p(x, t) = ∫Q_p(t - x)

    q_m(x, t) = Q_m(t + x)
    q_p(x, t) = Q_p(t - x)

    u_m(x, t) = u0(x - t) + ∫q_m(x, t) / 2
    v_m(x, t) = (q_m(x, t) + w_m(x, t)) / 2
    σ_m(x, t) = (q_m(x, t) - w_m(x, t)) / 2

    u_p(x, t) = u0(x + t) + ∫q_p(x, t) / 2
    v_p(x, t) =  (q_p(x, t) + w_p(x, t)) / 2
    σ_p(x, t) = -(q_p(x, t) - w_p(x, t)) / 2

    return u_m, u_p, v_m, v_p, σ_m, σ_p
end

function noncharacteristic_operator!(dq, q, D, b0, bN, HI, F)
    Nq = size(D, 1)
    rngs = ntuple(j-> ((j-1) * Nq + 1) : (j * Nq), 4)
    v_m, v_p, u_m, u_p = view.(Ref(q), rngs)

    V_m = v_p[1] - v_m[Nq]
    us_m, us_p = u_m[Nq], u_p[1]
    τs_m = F(V_m)
    τs_p = -τs_m

    operator!(dq, q, D, b0, bN, HI, τs_m, τs_p, us_m, us_p)
    nothing
end

function characteristic_operator!(dq, q, D, b0, bN, HI, F, γ)
    Nq = size(D, 1)
    rngs = ntuple(j-> ((j-1) * Nq + 1) : (j * Nq), 4)
    v_m, v_p, u_m, u_p = view.(Ref(q), rngs)

    us_m, us_p = q[4Nq + 1], q[4Nq + 2]

    # Normal
    n0, nN = -1, 1

    τ_m = nN * bN' * u_m + γ * (us_m - u_m[Nq])
    τ_p = n0 * b0' * u_p + γ * (us_p - u_p[ 1])

    w_m = v_m[Nq] - τ_m
    w_p = v_p[ 1] - τ_p

    η_m = w_p - w_m
    g(V_m) = 2F(V_m) + V_m - η_m
    V_m = bisection(g, zero(η_m), η_m)
    τs_m = F(V_m)
    τs_p = -τs_m

    operator!(dq, q, D, b0, bN, HI, τs_m, τs_p, us_m, us_p)

    dq[4Nq + 1] = w_m + τs_m
    dq[4Nq + 2] = w_p + τs_p

    nothing
end

function operator!(dq, q, D, b0, bN, HI, τs_m, τs_p, us_m, us_p)
    Nq = size(D, 1)
    rngs = ntuple(j-> ((j-1) * Nq + 1) : (j * Nq), 4)
    dv_m, dv_p, du_m, du_p = view.(Ref(dq), rngs)
    v_m, v_p, u_m, u_p = view.(Ref(q), rngs)

    # Set the displacement rates
    du_m .= v_m
    du_p .= v_p

    # Set the velocity rates
    dv_m .= D * u_m
    dv_p .= D * u_p

    # Normal
    n0, nN = -1, 1

    # Set the Neumann bcs
    dv_m[ 1] -= n0 * HI[ 1, 1] * b0' * u_m
    dv_p[Nq] -= nN * HI[Nq,Nq] * bN' * u_p

    # Set the interface values
    dv_m[Nq] += HI[Nq, Nq] * (τs_m - nN * bN' * u_m)
    dv_p[ 1] += HI[ 1,  1] * (τs_p - n0 * b0' * u_p)
    dv_m .-= nN * HI * bN * (us_m - u_m[Nq])
    dv_p .-= n0 * HI * b0 * (us_p - u_p[ 1])
    nothing
end

function convergence_test(p, Ns, β, cfl_c, cfl_nc, output = false)
    T = Float64

    x_m = range(T(-1), stop = 0, length=100)
    x_p = range(T(0), stop = 1, length=100)

    # Pulse center
    μ = -1 / 2

    # Pulse width
    w = 1 / 15

    u0(x) = 0.9 * exp(-((x - μ) / w)^2)
    v0(x) = derivative(t->u0(x-t), 0)
    σ0(x) = derivative(x->u0(x), x)

    tend::T = 1


    F(V) = β * asinh(V)

    ue_m, ue_p, ve_m, ve_p, σe_m, σe_p = semianalytic_solution(u0, v0, σ0, tend, F)

    # storage for error
    err_nc  = zeros(length(Ns))
    err_c   = zeros(length(Ns))

    for (iter, N) = enumerate(Ns)
        if output
            @info @sprintf " level %d with N = %4d" iter N
        end
        Nq = N + 1

        # Setup the SBP operators
        (D, S0, SN, HI, H, x_p) = DiagonalSBP.D2(p, N; xc = (0, 1))
        b0, bN = S0[1, :], SN[end, :]
        θ_R = DiagonalSBP.D2_remainder_parameters(p).θ_R
        θ_H = DiagonalSBP.D2_remainder_parameters(p).θ_H
        γ = N * (1 / θ_R + 1 / θ_H)

        # Create the grid
        x_m = x_p .- 1
        x = [x_m; x_p]

        # Create the solution
        q0 = zeros(T, 4Nq)
        q0[1:2Nq] = v0.(x)
        q0[2Nq .+ (1:2Nq)] = u0.(x)
        qnc = copy(q0)

        fnc!(Δq, q, _, _) = noncharacteristic_operator!(Δq, q, D, b0, bN, HI, F)

        qc = [q0; u0(0); u0(0)]
        fc!(Δq, q, _, _) = characteristic_operator!(Δq, q, D, b0, bN, HI, F, γ)

        h = 1 / N
        dtnc = cfl_nc * h
        dtc = cfl_c * h

        timestep!(qnc, fnc!, nothing, dtnc, (T(0), tend))
        timestep!(qc, fc!, nothing, dtc, (T(0), tend))

        rngs = ntuple(j-> ((j-1) * Nq + 1) : (j * Nq), 4)
        _, _, unc_m, unc_p = view.(Ref(qnc), rngs)
        _, _, uc_m, uc_p = view.(Ref(qc), rngs)
        err_c[iter] = sqrt(sum(H * ((uc_m - ue_m.(x_m, tend)).^2 +
                                    (uc_p - ue_p.(x_p, tend)).^2)))
        err_nc[iter] = sqrt(sum(H * ((unc_m - ue_m.(x_m, tend)).^2 +
                                     (unc_p - ue_p.(x_p, tend)).^2)))
        if output
            if iter > 1
                rate = ((log(err_nc[iter-1]) - log(err_nc[iter])) /
                        (log(Ns[iter]) - log(Ns[iter-1])))
                @info @sprintf """ Non-Characteristic
                error = %.2e
                rate  = %.2e""" err_nc[iter] rate
            else
                @info @sprintf """ Non-Characteristic
                error = %.2e""" err_nc[iter]
            end
            if iter > 1
                rate = ((log(err_c[iter-1]) - log(err_c[iter])) /
                        (log(Ns[iter]) - log(Ns[iter-1])))
                @info @sprintf """ Characteristic
                error = %.2e
                rate  = %.2e""" err_c[iter] rate
            else
                @info @sprintf """ Characteristic
                error = %.2e""" err_c[iter]
            end
        end
    end
    return err_c, err_nc
end

let
    Ns = 17 * 2 .^ (0:5)

    hs = 1 ./ Ns
    tri_x = [hs[end-1], hs[end], hs[end-1], hs[end-1]]
    tri_y = [hs[end-1], hs[end], hs[end  ], hs[end-1]] ./ hs[end-1]

    cfl_nc = Dict()
    cfl_nc[2, 1] = 1 // 2
    cfl_nc[4, 1] = 1 // 2
    cfl_nc[6, 1] = 1 // 2

    cfl_nc[2, 2] = 1 // 2
    cfl_nc[4, 2] = 1 // 4
    cfl_nc[6, 2] = 1 // 4

    cfl_nc[2, 32] = 1 // 32
    cfl_nc[4, 32] = 1 // 64
    cfl_nc[6, 32] = 1 // 64

    cfl_nc[2, 64] = 1 // 64
    cfl_nc[4, 64] = 1 // 128
    cfl_nc[6, 64] = 1 // 128

    cfl_nc[2, 128] = 1 // 128
    cfl_nc[4, 128] = 1 // 256
    cfl_nc[6, 128] = 1 // 256
    for β in (32, 64, 128)
        @show β

        C2, NC2 = convergence_test(2, Ns, β, 1 // 2, cfl_nc[2, β])
        C4, NC4 = convergence_test(4, Ns, β, 1 // 2, cfl_nc[4, β])
        C6, NC6 = convergence_test(6, Ns, β, 1 // 4, cfl_nc[6, β])

        @pgf a = LogLogAxis(
                            {
                             ymax = 1,
                             ymin = 1e-11,
                             xmin = hs[end]/2,
                             xmax = 2hs[1],
                             xlabel = "\$h\$",
                             ylabel = "\$\\|\\Delta u\\|_{H}\$",
                            }
                           )
        tc2 = 0.6*NC2[end-1] * tri_y.^2
        @pgf push!(a, Plot(
                           {
                            color = "black",
                           },
                           Table(tri_x, tc2)
                          )
                  )
        tc4 = 0.6*NC4[end-1] * tri_y.^4
        @pgf push!(a, Plot(
                           {
                            color = "black",
                           },
                           Table(tri_x, 0.6*NC4[end-1] * tri_y.^4)
                          )
                  )
        tc6 = 0.6*NC6[end-1] * tri_y.^6
        @pgf push!(a, Plot(
                           {
                            color = "black",
                           },
                           Table(tri_x, 0.6*NC6[end-1] * tri_y.^5)
                          )
                  )
        #=
        @pgf push!(a,
        [@sprintf("""
        \\node[anchor=north] () at (axis cs:%e, %e){\$s = 2\$};
        \\node[anchor=north] () at (axis cs:%e, %e){\$s = 4\$};
        \\node[anchor=north] () at (axis cs:%e, %e){\$s = 5\$};
        """,
        (hs[end-1] + hs[end]) / 2, tc2[2],
        (hs[end-1] + hs[end]) / 2, tc4[2],
        (hs[end-1] + hs[end]) / 2, tc6[2],
        )])
        =#
        @pgf push!(a,
                   [@sprintf("""
                             \\node[anchor=west] () at (axis cs:%e, %e){\$2\$};
                             \\node[anchor=west] () at (axis cs:%e, %e){\$4\$};
                             \\node[anchor=west] () at (axis cs:%e, %e){\$5\$};
                             """,
                             hs[end-1], exp((log(tc2[1]) + log(tc2[2]))/2),
                             hs[end-1], exp((log(tc4[1]) + log(tc4[2]))/2),
                             hs[end-1], exp((log(tc6[1]) + log(tc6[2]))/2),
                            )])

        @pgf push!(a, Plot(
                           {
                            color = "red",
                            mark = "o",
                           },
                           Table(hs, C2)
                          )
                  )
        @pgf push!(a, Plot(
                           {
                            color = "red",
                            mark = "x",
                           },
                           Table(hs, NC2)
                          )
                  )
        @pgf push!(a, Plot(
                           {
                            color = "blue",
                            mark = "o",
                           },
                           Table(hs, C4)
                          )
                  )
        @pgf push!(a, Plot(
                           {
                            color = "blue",
                            mark = "x",
                           },
                           Table(hs, NC4)
                          )
                  )
        @pgf push!(a, Plot(
                           {
                            color = "green",
                            mark = "o",
                           },
                           Table(hs, C6)
                          )
                  )
        @pgf push!(a, Plot(
                           {
                            color = "green",
                            mark = "x",
                           },
                           Table(hs, NC6)
                          )
                  )
        pgfsave("convergence_interface_$β.tex", a)
        pgfsave("convergence_interface_$β.pdf", a)
    end
end
