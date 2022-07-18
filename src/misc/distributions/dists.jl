using Distributions


const _EPS = 1e-7


class _TTransform(torch.distributions.Transform):
    
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    
    def _call(self, x):
        t = x[..., 0].unsqueeze(-1)
        v = x[..., 1:]
        return torch.cat((t, v * torch.sqrt(torch.clamp(1 - t ** 2, _EPS))), -1)

    def _inverse(self, y):
        t = y[..., 0].unsqueeze(-1)
        v = y[..., 1:]
        return torch.cat((t, v / torch.sqrt(torch.clamp(1 - t ** 2, _EPS))), -1)

    def log_abs_det_jacobian(self, x, y):
        t = x[..., 0]
        return ((x.shape[-1] - 3) / 2) * torch.log(torch.clamp(1 - t ** 2, _EPS))


class _HouseholderRotationTransform(torch.distributions.Transform):
    
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    
    def __init__(self, loc):
        super().__init__()
        self.loc = loc
        self.e1 = torch.zeros_like(self.loc)
        self.e1[..., 0] = 1

    def _call(self, x):
        u = self.e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + _EPS)
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def _inverse(self, y):
        u = self.e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + _EPS)
        return y - 2 * (y * u).sum(-1, keepdim=True) * u

    def log_abs_det_jacobian(self, x, y):
        return 0




class MarginalTDistribution(torch.distributions.TransformedDistribution):

    arg_constraints = {
        "dim": torch.distributions.constraints.positive_integer,
        "scale": torch.distributions.constraints.positive,
    }

    has_rsample = True

    def __init__(self, dim, scale, validate_args=None):
        self.dim = dim if isinstance(dim, torch.Tensor) else torch.tensor(dim, device=scale.device)
        self.scale = scale
        super().__init__(
            torch.distributions.Beta(
                (dim - 1) / 2 + scale, (dim - 1) / 2, validate_args=validate_args
            ),
            transforms=torch.distributions.AffineTransform(loc=-1, scale=2),
        )
        

    def entropy(self):
        return self.base_dist.entropy() + math.log(2)

    @property
    def mean(self):
        return 2 * self.base_dist.mean - 1

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        return 4 * self.base_dist.variance



struct PowerSpherical{T<:Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T
    marginal_t::Beta
    marginal_s::HyperSphericalUniform
end

function PowerSpherical(μ::Vector, κ)
    return PowerSpherical(
        μ, κ
        Beta(length(μ), κ),
        HyperSphericalUniform(length(μ) - 1)
    )
end

        
function log_prob(d::PowerSpherical, x):
    return log_normalizer(d) + d.κ * sum(log1p(d.μ .* x, dims = -1))

function log_normalizer(d::PowerSpherical):
        alpha, beta = params(d.marginal_t)

        return -(
            (alpha + beta) * log(2.) +
            lgamma(alpha) -
            lgamma(alpha + beta) +
            beta * log(pi)
        )

function entropy(d::PowerSpherical):
    alpha, beta = params(d.marginal_t)
    return -(
        log_normalizer(d) +
        d.κ *
        (log(2) + digamma(alpha) - digamma(alpha + beta))
    )

function mean(d::PowerSpherical):
    return d.μ * mean(d.marginal_t)

function stddev(d::PowerSpherical):
    return sqrt(variance(d))

function variance(d::PowerSpherical):
    alpha, beta = params(d.marginal_t)
    ratio = (alpha + beta) / (2 * beta)

    return variance(d.marginal_t) * (
        (1 - ratio) .* self.loc.unsqueeze(-1) @ self.loc.unsqueeze(-2)
        + ratio * torch.eye(self.loc.shape[-1])
    )


# @register_kl(PowerSpherical, HypersphericalUniform)
# def _kl_powerspherical_uniform(p, q):
#     return -p.entropy() + q.entropy()