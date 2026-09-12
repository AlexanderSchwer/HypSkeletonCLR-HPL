import torch

import geoopt as gt


GEOMETRY_MODELS = ("poincare", "lorentz")


class HyperbolicGeometry:
    """Small adapter around the hyperbolic models used by the training code."""

    def __init__(self, model="poincare", curvature=1.0):
        self.model = normalize_geometry_model(model)
        self.curvature = float(curvature)
        if self.curvature <= 0:
            raise ValueError("curvature must be positive")

        if self.model == "poincare":
            self.manifold = gt.PoincareBall(c=self.curvature)
        elif self.model == "lorentz":
            self.manifold = gt.Lorentz(k=self.curvature)
        else:
            raise ValueError(f"Unsupported hyperbolic geometry model: {model!r}")

    def expmap0(self, tangent):
        if self.model == "lorentz":
            tangent = self._lorentz_tangent(tangent)
        return self.manifold.projx(self.manifold.expmap0(tangent))

    def project(self, point):
        return self.manifold.projx(point)

    def dist(self, x, y):
        return self.manifold.dist(x, y)

    def dist0(self, x):
        return self.manifold.dist0(x)

    def tangent_proto_to_manifold(self, proto_tan):
        proto_norm = proto_tan.norm(dim=1, keepdim=True).clamp_min(1e-12)
        proto_tan = proto_tan * (torch.tanh(proto_norm) / proto_norm)
        return self.expmap0(proto_tan)

    def _lorentz_tangent(self, tangent):
        time = torch.zeros(
            *tangent.shape[:-1],
            1,
            dtype=tangent.dtype,
            device=tangent.device,
        )
        return torch.cat([time, tangent], dim=-1)


def make_hyperbolic_geometry(model="poincare", curvature=1.0):
    return HyperbolicGeometry(model=model, curvature=curvature)


def normalize_geometry_model(model):
    value = str(model or "poincare").strip().lower().replace("_", "-")
    aliases = {
        "poincare-ball": "poincare",
        "poincareball": "poincare",
        "geoopt": "poincare",
        "hyperboloid": "lorentz",
    }
    value = aliases.get(value, value)
    if value not in GEOMETRY_MODELS:
        raise ValueError(
            "Unsupported hyperbolic geometry model {!r}; expected one of {}".format(
                model,
                ", ".join(GEOMETRY_MODELS),
            )
        )
    return value
