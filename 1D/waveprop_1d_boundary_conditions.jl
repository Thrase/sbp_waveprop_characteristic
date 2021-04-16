using DiagonalSBP
using SparseArrays: sparsevec, spzeros, sparse
using LinearAlgebra: I, eigvals
using Logging: @info
using Printf: @sprintf

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

  Γ = N / DiagonalSBP.D2_remainder_parameters(p).β

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
  f(x) = exp(-((x - μ) / w)^2)
  fp(x) = -2((x - μ) / w^2) * exp(-((x - μ) / w)^2)

  ue(x, t) = f(x - t) + R * f(2 - x - t)
  ue_t(x, t) = -fp(x - t) - R * fp(2 - x - t)
  ue_x(x, t) =  fp(x - t) - R * fp(2 - x - t)

  # storage for error
  err_NC  = zeros(length(Ns))
  err_C   = zeros(length(Ns))

  t_final = 1

  # loop over all N
  for (iter, N) = enumerate(Ns)
    @info @sprintf " level %d with N = %4d" iter N

    (NC, x, H) = noncharacteristic_operator(p, N, R)
    q0 = [ue.(x, 0); ue_t.(x, 0)]

    qf = exp(t_final * Matrix(NC)) * q0
    qf_ex = [ue.(x, 1); ue_t.(x, 1)]

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

    (NC, x, H) = characteristic_operator(p, N, R)
    q0 = [ue.(x, 0); ue_t.(x, 0); 0; 0]

    qf = exp(t_final * Matrix(NC)) * q0
    qf_ex = [ue.(x, 1); ue_t.(x, 1); ue(0, 1); ue(1, 1)]

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
