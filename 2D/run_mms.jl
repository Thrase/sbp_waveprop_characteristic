using SparseArrays
using LinearAlgebra
using SparseArrays
using ForwardDiff: derivative

include("sim_funs.jl")
include("square_circle.jl")

# Define the mms solution
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

# Run the convergence results
function main(
    sbp_order,
    refinement_levels,
    N0;
    characteristic_method = true,
    cfl = 2,
    friction = (V) -> asinh(V),
    tspan = (0.0, 1),
    do_output = true
)
    # Define the friction law to use

    # The element to domain map is needed here in order to define the boundary
    # functions (we have a circular argument going on that we need EToDomain to
    # evaluate the boundary conditions, but we also need the boundary conditions
    # to build the square_circle operators)
    bc_map = [
        BC_DIRICHLET,
        BC_DIRICHLET,
        BC_NEUMANN,
        BC_NEUMANN,
        characteristic_method ? BC_JUMP_INTERFACE : -BC_JUMP_INTERFACE,
    ]
    if do_output
      @show bc_map
      @show [BC_DIRICHLET, BC_NEUMANN, BC_JUMP_INTERFACE]
    end

    (_, _, _, _, EToDomain) = read_inp_2d("square_circle.inp"; bc_map = bc_map)

    # Boundary condition function defined from the mms solution
    gDfun(x, y, t, e) = ue(x, y, t, EToDomain[e])
    gDdotfun(x, y, t, e) = ∂t_ue(x, y, t, EToDomain[e])
    function gNfun(nx, ny, xf, yf, t, e)
        dom = EToDomain[e]
        return nx .* (∂x_ue(xf, yf, t, dom)) + ny .* (∂y_ue(xf, yf, t, dom))
    end

    body_force(x, y, t, e) = force(x, y, t, EToDomain[e])

    # Loop over the levels of refinement
    ϵ = zeros(refinement_levels)
    for lvl in 1:length(ϵ)
        # Set up the local grid dimensions
        Nr = N0 * (2^(lvl - 1))
        Ns = N0 * (2^(lvl - 1))

        Nqr = Nr + 1
        Nqs = Ns + 1

        Np = Nqr * Nqs

        # Create the operators for the problem
        (metrics, rhsops, EToDomain, EToF, EToO, EToS, FToB, FToE, FToLF) =
            build_square_circle(
                sbp_order,
                Nqr,
                Nqs,
                gDfun,
                gDdotfun,
                gNfun,
                body_force,
                bc_map,
                (lvl == 1) && do_output,
            )
        nelems = length(rhsops)

        # initial conditions
        lenq0 = 2 * Np + 2 * Nqr + 2 * Nqs
        q = Array{Float64, 1}(undef, nelems * lenq0)
        for e in 1:nelems
            (xf1, xf2, xf3, xf4) = metrics[e].facecoord[1]
            (yf1, yf2, yf3, yf4) = metrics[e].facecoord[2]

            dom = EToDomain[e]
            u0 = ue.(metrics[e].coord[1][:], metrics[e].coord[2][:], 0, dom)
            v0 = ∂t_ue.(metrics[e].coord[1][:], metrics[e].coord[2][:], 0, dom)
            û10 = ue.(xf1[:], yf1[:], 0, dom)
            û20 = ue.(xf2[:], yf2[:], 0, dom)
            û30 = ue.(xf3[:], yf3[:], 0, dom)
            û40 = ue.(xf4[:], yf4[:], 0, dom)

            q[(lenq0 * (e - 1) + 1):(e * lenq0)] =
                [u0; v0; û10; û20; û30; û40]
        end

        # solve the ODE
        hmin = mapreduce(m -> m.hmin, min, values(metrics))
        dt = cfl * hmin
        if do_output
          @show dt
        end
        params = (
            Nqr = Nqr,
            Nqs = Nqs,
            rhsops = rhsops,
            EToF = EToF,
            EToO = EToO,
            EToS = EToS,
            FToB = FToB,
            FToE = FToE,
            FToLF = FToLF,
            friction = friction,
        )
        timestep!(q, waveprop!, params, dt, tspan)

        # Loop over the blocks and compute the error in each block
        for e in 1:nelems
            qe = @view q[(lenq0 * (e - 1) + 1):(e * lenq0)]

            qe_u = qe[1:Np]

            dom = EToDomain[e]
            qexact_u =
                ue.(
                    metrics[e].coord[1][:],
                    metrics[e].coord[2][:],
                    tspan[2],
                    dom,
                )
            qexact_v =
                ∂t_ue.(
                    metrics[e].coord[1][:],
                    metrics[e].coord[2][:],
                    tspan[2],
                    dom,
                )
            qexact_û1 =
                ue.(
                    metrics[e].facecoord[1][1],
                    metrics[e].facecoord[2][1],
                    tspan[2],
                    dom,
                )
            qexact_û2 =
                ue.(
                    metrics[e].facecoord[1][2],
                    metrics[e].facecoord[2][2],
                    tspan[2],
                    dom,
                )
            qexact_û3 =
                ue.(
                    metrics[e].facecoord[1][3],
                    metrics[e].facecoord[2][3],
                    tspan[2],
                    dom,
                )
            qexact_û4 =
                ue.(
                    metrics[e].facecoord[1][4],
                    metrics[e].facecoord[2][4],
                    tspan[2],
                    dom,
                )

            Δu = qe_u - qexact_u
            Mv = rhsops[e].JH

            ϵ[lvl] += Δu' * Mv * Δu
        end # end compute error at lvl

        ϵ[lvl] = sqrt(ϵ[lvl])
        if do_output
          @show (lvl, ϵ[lvl])
          if lvl > 1
            rate = (log.(ϵ[lvl-1]) - log.(ϵ[lvl])) / log(2)
            @show rate
          end
          println()
        end
    end #loop over levels
    return ϵ
end
