using ForwardDiff: derivative

function ue(x, y, t, dom, A1 = 5 * exp(1) / (1 + exp(1)), A2 = 5)
  r = hypot(x, y)
  θ = atan(y, x)
  if dom == 1
    return A1 * sin(t) * (1 - exp(-1 * r^2)) * r * sin(θ)
  else
    return A2 * sin(t) * ((r - 1)^2 * cos(θ) + (r - 1) * sin(θ))
  end
end

∂t_ue(x, y, t, dom) = derivative(t -> ue(x, y, t, dom), t)
∂tt_ue(x, y, t, dom) = derivative(t -> ∂t_ue(x, y, t, dom), t)

∂x_ue(x, y, t, dom) = derivative(x -> ue(x, y, t, dom), x)
∂y_ue(x, y, t, dom) = derivative(y -> ue(x, y, t, dom), y)

∂xx_ue(x, y, t, dom) = derivative(x -> ∂x_ue(x, y, t, dom), x)
∂yy_ue(x, y, t, dom) = derivative(y -> ∂y_ue(x, y, t, dom), y)

function force(x, y, t, dom)
    return ∂tt_ue(x, y, t, dom) - (∂xx_ue(x, y, t, dom) + ∂yy_ue(x, y, t, dom))
end
