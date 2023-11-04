using Optim

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
optimize(f, [0.0, 0.0])

function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    end

lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
inner_optimizer = GradientDescent()
results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
results = optimize(f, g!, lower, upper, initial_x, Fminbox(GradientDescent()), Optim.Options(outer_iterations = 2))
