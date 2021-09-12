import MLJModelInterface
using MLJModelInterface: Model
import MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass
const MMI = MLJModelInterface
using CategoricalArrays
import MLJBase


"""
Compliance type RFE ,which we build to select feature, can be 
used for both Probabilistic and Deterministic.
"""
mutable struct RFE{M<:Model} <: MMI.Supervised
    model::M
    n_features_to_select::Integer
    nstep::Integer
end

"""
```julia
    RFE(model::M; n_features_to_select=3, nstep=1)
```
Constructor of composite type RFE.

# parameter

* `model::MLJModelInterface.Model`: 
        For example: logistic regression
        @load LogisticClassifier pkg=MLJLinearModels
        model = LogisticClassifier()

* `n_features_to_select`: The number of features to select. int, default = 3

* `nstep`: the number of features to remove at each iteration. int, default = 1
"""
function RFE(model::M; n_features_to_select=3, nstep=1) where M<:Model
    model   = RFE(model, n_features_to_select, nstep)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


"""
```julia
    MMI.clean!(m::RFE)
```
Overload the clean! method of MLJModelInterface

# parameter

* `m::RFE`: Compliance type RFE
"""
function MMI.clean!(m::RFE)
    warning = ""
    if m.n_features_to_select < 0
        warning *= "Parameter `n_features_to_select` expected to be positive, resetting to 1"
        m.n_features_to_select = 1

    elseif m.n_features_to_select == 0
        warning *= "Find Parameter `n_features_to_select` is zero, so resetting to number of features divided by 2"
        m.n_features_to_select = div(n_features,2)
    end

    if m.nstep <= 0
        warning *= "Parameter `nstep` expected to be positive, resetting to 1"
        m.nstep = 1
    end
    return warning
end


"""
```julia
    MMI.fit(rfemodel::RFE, verbosity, X, y)
```
Overload the fit method of MLJModelInterface

# parameter

* `rfemodel::RFE`: Compliance type RFE

* `X`: sample data, MLJModelInterface.tabel(Continuous, Count) format

* 'y': sample label, AbstractVector{MMI.Continuous} format
"""
function MMI.fit(rfemodel::RFE, verbosity, X, y)

    # parse the parameters from rfemodel
    kwargs = Tuple{Symbol, Any}[]
    m = []
    for fn in fieldnames(typeof(rfemodel))
        if fn == :model
            push!(m, getfield(rfemodel, fn))
        elseif fn != :fit
                push!(kwargs, (fn, getfield(rfemodel, fn)))
        end
    end
    # model training
    support_, ranking_ = rfe(X, y, m[1]; kwargs...)

    fitresult = NamedTuple{(:support_, :ranking_)}((support_, ranking_))
    cache=nothing
    report=nothing
    return fitresult, cache, report
end


"""
```julia
    rfe(X, y, m; args...)
```
    internal function，invoked by fit

# parameter

* `X`: sample data, MLJModelInterface.tabel(Continuous, Count) format

* 'y': sample label, AbstractVector{MMI.Continuous} format

* `m::Model`: any model in MLJ
            For example: logistic regression
            @load LogisticClassifier pkg=MLJLinearModels
            m = LogisticClassifier()

* 'args': hyperparameters for rfe, include n_features_to_select and nstep
"""
function rfe(X, y, m; args...)

    n_features_to_select = args[1]
    step1 = args[2]
    
    n_features = length(X);
    
    support_ = ones(Bool, n_features);
    ranking_ = ones(Int32, n_features);
    

    while  sum(support_) > n_features_to_select

        features = [i for i=1:n_features if support_[i]==true]

        # model training
        clf = train(m, MMI.selectcols(X, features), y)
        # get feature importances based on model coefficients
        importances = get_feature_importances(clf, transform_func="square")

        feature_importance = Dict()
        for i in 1:length(features)
            global feature_importance[features[i]] = importances[i]
        end

        ranks = sort(importances)
        threshold = min(step1, sum(support_) - n_features_to_select)
        falserank=ranks[1:threshold]
        for (index, item) in enumerate(feature_importance)
            if(item[2] in falserank)
                    support_[item[1]] = false
            end
        end
        
        for i in 1:length(support_)
            if (support_[i] == false)
                ranking_[i] += 1
            end
        end
    end
    return (support_, ranking_)
end


"""
```Julia
    train(m, X, y)
```
invoke MLJ's machine and fit!, train model m

# Parameters

* `m::Model`: any model in MLJ
        For example: logistic regression
        @load LogisticClassifier pkg=MLJLinearModels
        m = LogisticClassifier()
        
* `X`: sample data, MLJModelInterface.tabel(Continuous, Count) format

* 'y': sample label, AbstractVector{MMI.Continuous} format
"""
function train(m, X, y)
    clf = machine(m, X, y)
    fit!(clf)
    return clf
end


"""
```Julia
    get_feature_importances(clf; transform_func="None")
```
Retrieve and aggregate (ndim > 1)  the feature importances from an model.

# Parameters

* `clf`:  A MLJ estimator from which we want to get the feature importances.
        
* `transform_func` : {"norm", "square"}, default=None
                    The transform to apply to the feature importances. By default (`None`)
                    no transformation is applied.
"""
function get_feature_importances(clf; transform_func="None")
    params = fitted_params(clf)
    if :coefs in fieldnames(typeof(params))
        coefs = params.coefs
    elseif :importances in fieldnames(typeof(params))
            coefs = params.importances
    else
        throw("the model putted in RFE has no attribute about coefs or importances")
    end
    
    dim = length(coefs[1].second)
    importances = []

    if transform_func=="None"
        if dim == 1
            for i in coefs
                push!(importances, i.second)
            end  
        else
            for i in coefs
                push!(importances, sum(i.second))
            end
        end

    elseif transform_func == "norm"
        if dim == 1
            for i in coefs
                push!(importances, abs.(i.second))
            end  
        else
            for i in coefs
                push!(importances, sum(abs.(i.second)))
            end
        end

    elseif transform_func == "square"
        if dim == 1
            for i in coefs
                push!(importances, abs2.(i.second))
            end  
        else
            for i in coefs
                push!(importances, sum(abs2.(i.second)))
            end
        end
    end

    return importances
end

"""
metadata configuration:
"""
## Restrict input sample format
MMI.input_scitype(::Type{<:RFE}) = MMI.Table(MMI.Continuous, MMI.Count)
## Restrict sample label format
MMI.target_scitype(::Type{<:RFE}) = AbstractVector{MMI.Continuous}
## whether it is pure julia code
MMI.is_pure_julia(::Type{<:RFE}) = true

"""
The following configuration can not be finished until our code is open source to github.
"""
# MMI.load_path(::Type{<:RFE}) = "" # lazy-loaded from MLJ
# MMI.package_name(::Type{<:RFE}) = ""
# MMI.package_uuid(::Type{<:RFE}) = ""
# MMI.package_url(::Type{<:RFE}) = "https://github.com/xxx/xxx.jl"
