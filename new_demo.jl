include("./rfe.jl")
using MLJ
import MLJModelInterface


# A simple regressor
@load HuberRegressor pkg=MLJLinearModels
X, y = @load_boston
mdl = HuberRegressor()
demo_m = RFE(mdl; n_features_to_select=5, nstep=1);
mach = machine(demo_m, X, y)
fit!(mach)
println(mach.fitresult)

# A simple  classifier
@load MultinomialClassifier pkg=MLJLinearModels
X, y = @load_iris
mdl = MultinomialClassifier(lambda=0.5, gamma=0.7)
demo_m = RFE(mdl; n_features_to_select=2, nstep=1);
mach = machine(demo_m, X, y)
fit!(mach)
println(mach.fitresult)