function convergence_test(c1, c2)
  # Pulse center
  μ = -1 / 2

  # Pulse width
  w = 1 / 15

  # Analytic solution
  f(x) = exp(-((x - μ) / w)^2)
  fp(x) = -2((x - μ) / w^2) * exp(-((x - μ) / w)^2)

  u1(x, t) = f(x - c1 * t) + ((c2 - c1) / (c1 + c2)) * f(-x - c1 * t)
  u2(x, t) = (2c2 / (c1 + c2)) * f(x * c1 / c2 - c1 * t)

  u1_x(x, t) = fp(x - c1 * t) - ((c2 - c1) / (c1 + c2)) * fp(-x - c1 * t)
  u2_x(x, t) = (2c1 / (c1 + c2)) * fp(x * c1 / c2 - c1 * t)

  for t in range(0.0, stop = c1, length = 100)
    @assert u1(0, t) ≈ u2(0, t)
    @assert u1_x(0, t) ≈ u2_x(0, t)
  end
end
