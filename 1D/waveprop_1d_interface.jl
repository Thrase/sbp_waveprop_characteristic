using DiagonalSBP
using SparseArrays: sparsevec, spzeros, sparse
using LinearAlgebra: I, eigvals
using Logging: @info
using Printf: @sprintf

function noncharacteristic_operator(p, N, ρL, ρR, μL, μR)
  # Total number of points
  Nq = N + 1

  # Get the various derivative operators
  (D, S0, SN, HI, H, xR) = DiagonalSBP.D2(p, N; xc = (0, 1))
  xL = xR .- 1
  B¹, B² = S0[1:1, :], SN[Nq:Nq, :]

  # Form the SBP inner product matrix
  A = -H * D + SN - S0

  L¹ = sparse([1], [ 1], [1], 1, Nq)
  L² = sparse([1], [Nq], [1], 1, Nq)
  n¹ = -1
  n² = +1

  Γ = 4N / DiagonalSBP.D2_remainder_parameters(p).β

  # Solving the system in first order form with [u; v]
  # Block 1 (left)
  OuLuL = spzeros(Nq, Nq)
  OuLvL = I
  OuLuR = spzeros(Nq, Nq)
  OuLvR = spzeros(Nq, Nq)

  OvLuL = -μL * A
  OvLvL = spzeros(Nq, Nq)
  OvLuR = spzeros(Nq, Nq)
  OvLvR = spzeros(Nq, Nq)

  OvLuL += μL * L²' * (n² * B² - Γ * L²) / 2 + μL * n² * B²' * L² / 2
  OvLuR -= μR * L²' * (n¹ * B¹ - Γ * L¹) / 2 + μR * n² * B²' * L¹ / 2

  # Block 2 (right)
  OuRuL = spzeros(Nq, Nq)
  OuRvL = spzeros(Nq, Nq)
  OuRuR = spzeros(Nq, Nq)
  OuRvR = I

  OvRuL = spzeros(Nq, Nq)
  OvRvL = spzeros(Nq, Nq)
  OvRuR = -μR * A
  OvRvR = spzeros(Nq, Nq)

  OvRuL -= μL * L¹' * (n² * B² - Γ * L²) / 2 + μL * n¹ * B¹' * L² / 2
  OvRuR += μR * L¹' * (n¹ * B¹ - Γ * L¹) / 2 + μR * n¹ * B¹' * L¹ / 2

  # Form the operator
  O = [      OuLuL OuLvL OuLuR OuLvR
       HI * [OvLuL OvLvL OvLuR OvLvR] / ρL
             OuRuL OuRvL OuRuR OuRvR
       HI * [OvRuL OvRvL OvRuR OvRvR] / ρR]

  return (O, xL, xR, H)
end


function convergence_test(p, Ns, ρL, ρR, μL, μR)
  @info "sbp order = $p"

  # Calculate wave speeds
  cL = 1 / sqrt(μL / ρL)
  cR = 1 / sqrt(μR / ρR)

  # Pulse center
  center = -1 / 2

  # Pulse width
  w = 1 / 15

  # Analytic solution
  f(x) = exp(-((x - center) / w)^2)
  fp(x) = -2((x - center) / w^2) * exp(-((x - center) / w)^2)

  uL(x, t) = f(x - cL * t) + ((cR - cL) / (cL + cR)) * f(-x - cL * t)
  uR(x, t) = (2cR / (cL + cR)) * f(x * cL / cR - cL * t)

  uL_t(x, t) = -cL * fp(x - cL * t) + -cL * ((cR - cL) / (cL + cR)) * fp(-x - cL * t)
  uR_t(x, t) = -cL * (2cR / (cL + cR)) * fp(x * cL / cR - cL * t)

  uL_x(x, t) = fp(x - cL * t) - ((cR - cL) / (cL + cR)) * fp(-x - cL * t)
  uR_x(x, t) = (2cL / (cL + cR)) * fp(x * cL / cR - cL * t)

  # storage for error
  err_NC  = zeros(length(Ns))
  err_C   = zeros(length(Ns))

  t_final = 1

  for t in range(0.0, stop = cR, length = 100)
    @assert uL(0, t) ≈ uR(0, t)
    @assert uL_t(0, t) ≈ uR_t(0, t)
    @assert uL_x(0, t) ≈ uR_x(0, t)
  end

  for (iter, N) = enumerate(Ns)
    @info @sprintf " level %d with N = %4d" iter N

    (NC, xL, xR, H) = noncharacteristic_operator(p, N, ρL, ρR, μL, μR)
    q0 = [uL.(xL, 0); uL_t.(xL, 0); uR.(xR, 0); uR_t.(xR, 0)]

    qf = exp(t_final * Matrix(NC)) * q0
    qf_ex = [uL.(xL, 1); uL_t.(xL, 1); uR.(xR, 1); uR_t.(xR, 1)]

    ϵ = qf_ex - qf

    Nq = N + 1
    err_NC[iter] = sqrt(ϵ[1:Nq]' * H * ϵ[1:Nq] + ϵ[(2Nq+1):3Nq]' * H * ϵ[(2Nq+1):3Nq])
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
  end

end
