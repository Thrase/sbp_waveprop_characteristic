using DiagonalSBP
using SparseArrays: sparsevec, spzeros, sparse
using LinearAlgebra: I, eigvals
using Logging: @info
using Printf: @sprintf
using PGFPlotsX: @pgf, Plot, Axis, Table, pgfsave, SemiLogYAxis
using ForwardDiff: derivative

function noncharacteristic_operator(p, N, R)
  # Total number of points
  Nq = N + 1

  # Get the various derivative operators
  (D, S0, SN, HI, H, x) = DiagonalSBP.D2(p, N; xc = (0, 1))

  # Form the SBP inner product matrix
  A = -H * D + SN - S0

  L¹ = sparsevec([ 1], [1], Nq)'
  L² = sparsevec([Nq], [1], Nq)'

  α = -(1 - R) / (1 + R)

  # Solving the system in first order form with [u; v]
  Ouu = spzeros(Nq, Nq)
  Ouv = I
  Ovu = -A
  Ovv = α * L²' * L² + α * L¹' * L¹ 

  # Form the operator
  O = [Ouu Ouv;
       HI*Ovu HI*Ovv]

  return (O, x, H)
end

function characteristic_operator(p, N, R)
  # Total number of points
  Nq = N + 1

  # Get the various derivative operators
  (D, S0, SN, HI, H, x) = DiagonalSBP.D2(p, N; xc = (0, 1))
  B¹, B² = S0[1:1, :], SN[Nq:Nq, :]

  # Form the SBP inner product matrix
  A = -H * D + SN - S0

  L¹ = sparse([1], [ 1], [1], 1, Nq)
  L² = sparse([1], [Nq], [1], 1, Nq)
  n¹ = -1
  n² = +1

  θ_R = DiagonalSBP.D2_remainder_parameters(p).θ_R
  θ_H = DiagonalSBP.D2_remainder_parameters(p).θ_H
  Γ = N * (1 / θ_R + 1 / θ_H)

  # Solving the system in first order form with [u; v; ϕ¹; ϕ²]
  Ouu = spzeros(Nq, Nq)
  Ouv = I
  Ouϕ¹ = spzeros(Nq, 1)
  Ouϕ² = spzeros(Nq, 1)

  Ovu  = -A
  Ovv  = spzeros(Nq, Nq)
  Ovϕ¹ = spzeros(Nq, 1)
  Ovϕ² = spzeros(Nq, 1)

  Oϕ¹u  = spzeros(1, Nq)
  Oϕ¹v  = spzeros(1, Nq)
  Oϕ¹ϕ¹ = spzeros(1, 1)
  Oϕ¹ϕ² = spzeros(1, 1)

  Oϕ²u  = spzeros(1, Nq)
  Oϕ²v  = spzeros(1, Nq)
  Oϕ²ϕ¹ = spzeros(1, 1)
  Oϕ²ϕ² = spzeros(1, 1)

  Ovu  += ((1 - R) / 2) * L¹' * (n¹ * B¹ - Γ * L¹) + n¹ * B¹' * L¹
  Ovv  += -((1 - R) / 2) * L¹' * L¹
  Ovϕ¹ += ((1 - R) / 2) * Γ * L¹' - n¹ * B¹'

  Oϕ¹u  += ((1 + R) / 2) * (Γ * L¹ - n¹ * B¹)
  Oϕ¹v  += ((1 + R) / 2) * L¹
  Oϕ¹ϕ¹ += sparse([1], [1], [-Γ * (1 + R) / 2], 1, 1)

  Ovu  +=  ((1 - R) / 2) * L²' * (n² * B² - Γ * L²) + n² * B²' * L²
  Ovv  += -((1 - R) / 2) * L²' * L²
  Ovϕ² +=  ((1 - R) / 2) * Γ * L²' - n² * B²'

  Oϕ²u  += ((1 + R) / 2) * (Γ * L² - n² * B²)
  Oϕ²v  += ((1 + R) / 2) * L²
  Oϕ²ϕ² += sparse([1], [1], [-Γ * (1 + R) / 2], 1, 1)

  # Form the operator
  O = [    Ouu  Ouv  Ouϕ¹  Ouϕ²
       HI*[Ovu  Ovv  Ovϕ¹  Ovϕ²]
           Oϕ¹u Oϕ¹v Oϕ¹ϕ¹ Oϕ¹ϕ²
           Oϕ²u Oϕ²v Oϕ²ϕ¹ Oϕ²ϕ²]

  return (O, x, H)
end

function convergence_test(p, Ns, R)
  @info "sbp order = $p"

  # Pulse center
  μ = 1 / 2

  # Pulse width
  w = 1 / 15

  # Analytic solution
  f(x) = exp(-((x - μ) / w)^2) / 2

  ue(x, t) = (f(x - t) + f(x + t)) + R * (f(2 - x - t) + f(-x + t))
  ∂t_ue(x, t) = derivative(t->ue(x, t), t)

  # storage for error
  err_NC  = zeros(length(Ns))
  err_C   = zeros(length(Ns))

  t_final = 1

  # loop over all N
  for (iter, N) = enumerate(Ns)
    @info @sprintf " level %d with N = %4d" iter N

    (NC, x, H) = noncharacteristic_operator(p, N, R)
    q0 = [ue.(x, 0); ∂t_ue.(x, 0)]

    qf = exp(t_final * Matrix(NC)) * q0
    qf_ex = [ue.(x, 1); ∂t_ue.(x, 1)]

    ϵ = qf_ex - qf

    err_NC[iter] = sqrt(ϵ[1:N+1]' * H * ϵ[1:N+1])
    if iter > 1
      rate = ((log(err_NC[iter-1]) - log(err_NC[iter])) /
              (log(Ns[iter]) - log(Ns[iter-1])))
      @info @sprintf """ Non-Characteristic
      error = %.2e
      rate  = %.2e""" err_NC[iter] rate
    else
      @info @sprintf """ Non-Characteristic
      error = %.2e""" err_NC[iter]
    end

    (C, x, H) = characteristic_operator(p, N, R)
    q0 = [ue.(x, 0); ∂t_ue.(x, 0); 0; 0]

    qf = exp(t_final * Matrix(C)) * q0
    qf_ex = [ue.(x, 1); ∂t_ue.(x, 1); ue(0, 1); ue(1, 1)]

    ϵ = qf_ex - qf

    err_C[iter] = sqrt(ϵ[1:N+1]' * H * ϵ[1:N+1])
    if iter > 1
      rate = ((log(err_C[iter-1]) - log(err_C[iter])) /
              (log(Ns[iter]) - log(Ns[iter-1])))
      @info @sprintf """ Characteristic
      error = %.2e
      rate  = %.2e""" err_C[iter] rate
    else
      @info @sprintf """ Characteristic
      error = %.2e""" err_C[iter]
    end
  end
end

function make_spectrum_plots(p, N, R)
  h = 1 / N
  (NC, _...) = noncharacteristic_operator(p, N, R)
  (C, _...)  = characteristic_operator(p, N, R)

  ev_NC = h * eigvals(Matrix(NC))
  ev_C  = h * eigvals(Matrix(C))

  ymin, ymax, xmin, xmax = -2.5, 2.5, -3.35, 1e-1

  pts = nothing
  if !all(xmin .< real.(ev_NC) .< xmax)
    pts = ev_NC[findall(.!(xmin .< real.(ev_NC) .< xmax))]
    @assert length(pts) == 2
    @assert pts[1] ≈ pts[2]
    @assert abs(imag(pts[1])) < 1e-14
    ev_NC = [ev_NC[xmin .< real.(ev_NC) .< xmax]; -3.1]
  end
  @assert all(xmin .< real.(ev_C ) .< xmax)
  @assert all(ymin .< imag.(ev_NC) .< ymax)
  @assert all(ymin .< imag.(ev_C ) .< ymax)

  xmin = xmin - 0.2

  @pgf a = Axis(
            {
             ymin = ymin,
             ymax = ymax,
             xmin = xmin,
             xmax = xmax,
             legend_entries = {"non-characteristic", "characteristic"},
             title = "R = $R, N = $N",
             xlabel = "\$\\textrm{Re}(h \\lambda)\$",
             ylabel = "\$\\textrm{Im}(h \\lambda)\$",
            }
           )
  @pgf push!(a, Plot(
                     {
                      only_marks,
                      color = "red",
                      mark = "x",
                     },
                     Table(real.(ev_NC), imag.(ev_NC)))
            )
  @pgf push!(a, Plot(
                     {
                      only_marks,
                      color = "blue",
                      mark = "+",
                     },
                     Table(real.(ev_C), imag.(ev_C))),
            )
  if !isnothing(pts)
    @pgf push!(a,
               [@sprintf """
                \\node[anchor=south] () at (axis cs:-3.1,0){\$%.2e\$};
                \\node[anchor=west] (source) at (axis cs:-3.1,0){};
                \\node (destination) at (axis cs:-3.35,0){};
                \\draw[->](source)--(destination);
                """ real(pts[1])])
  end

  pgfsave("bc_spectra_$R.tex", a)
  pgfsave("bc_spectra_$R.pdf", a)
end

function compare_eigs(p, N)
  h = 1 / N
  Rs = -0.95:0.05:0.95
  real_C = zeros(length(Rs))
  real_NC = zeros(length(Rs))
  imag_C = zeros(length(Rs))
  imag_NC = zeros(length(Rs))
  for (iter, R) = enumerate(Rs)
    (NC, _...) = noncharacteristic_operator(p, N, R)
    (C, _...)  = characteristic_operator(p, N, R)

    ev_NC = h * eigvals(Matrix(NC))
    ev_C  = h * eigvals(Matrix(C))

    real_C[iter] = minimum(real.(ev_C))
    real_NC[iter] = minimum(real.(ev_NC))
    imag_C[iter] = maximum(imag.(ev_C))
    imag_NC[iter] = maximum(imag.(ev_NC))
  end

  @pgf a = SemiLogYAxis(
                {
                 xmin = -1,
                 xmax =  1,
                 xlabel = "\$R\$",
                 ylabel = "max\$(-\\textrm{Re}(h \\lambda)\$",
                 legend_entries = {"non-characteristic", "characteristic"},
                }
               )
  @pgf push!(a, Plot(
                     {
                      only_marks,
                      color = "red",
                      mark = "x",
                     },
                     Table(Rs, -real_NC)
                    )
            )
  @pgf push!(a, Plot(
                     {
                      only_marks,
                      color = "blue",
                      mark = "+",
                     },
                     Table(Rs, -real_C)
                    )
            )
  pgfsave("real_eig.tex", a)
  pgfsave("real_eig.pdf", a)
end

for R in (-0.99, -0.9, 0, 0.9, 1.0)
  make_spectrum_plots(4, 50, R)
end
compare_eigs(4, 50)
