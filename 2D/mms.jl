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
