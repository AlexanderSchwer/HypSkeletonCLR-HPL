import torch
import torch.nn.functional as F

from tools.hyperbolic_geometry import make_hyperbolic_geometry


HIERARCHY_OBJECTIVES = ("ranking_ce", "similarity_weighted")


@torch.no_grad()
def prototype_affinity_hyp(proto_h, curvature, temperature=1.0, geometry_model="poincare"):
    """
    Compute pairwise prototype affinity from hyperbolic distance.
    """
    if proto_h.dim() != 2:
        raise ValueError(f"proto_h must be [K, D], got shape {tuple(proto_h.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    geometry = make_hyperbolic_geometry(geometry_model, curvature)
    pairwise_dist = geometry.dist(proto_h.unsqueeze(1), proto_h.unsqueeze(0))
    affinity = torch.exp(-pairwise_dist / temperature)
    affinity.fill_diagonal_(0.0)
    return affinity / affinity.sum().clamp_min(1e-12)


@torch.no_grad()
def update_affinity_ema(aff_ema, batch_aff, momentum=0.9):
    """
    Update cluster-cluster affinity with exponential moving average.

    Args:
        aff_ema: [K, K] or None
        batch_aff: [K, K] pairwise cluster affinity
        momentum: EMA factor
    Returns:
        [K, K] updated affinity
    """
    if batch_aff.dim() != 2 or batch_aff.shape[0] != batch_aff.shape[1]:
        raise ValueError(f"batch_aff must be square [K, K], got shape {tuple(batch_aff.shape)}")
    if not (0.0 <= momentum < 1.0):
        raise ValueError("momentum must be in [0, 1)")

    n_clusters = batch_aff.shape[0]
    batch_aff = batch_aff.clone()
    batch_aff = 0.5 * (batch_aff + batch_aff.t())
    batch_aff.fill_diagonal_(0.0)
    batch_aff = batch_aff / batch_aff.sum().clamp_min(1e-12)

    if aff_ema is None:
        return batch_aff.detach()

    if aff_ema.shape != (n_clusters, n_clusters):
        raise ValueError(
            f"aff_ema shape mismatch: expected {(n_clusters, n_clusters)}, got {tuple(aff_ema.shape)}"
        )

    updated = momentum * aff_ema + (1.0 - momentum) * batch_aff
    updated = 0.5 * (updated + updated.t())
    updated.fill_diagonal_(0.0)
    updated = updated / updated.sum().clamp_min(1e-12)
    return updated.detach()


@torch.no_grad()
def sample_triplets_from_affinity(aff, n_triplets):
    """
    Sample (anchor, positive, negative) from cluster affinity matrix.
    Higher affinity is preferred for positives and lower for negatives.
    """
    if aff is None:
        return torch.empty((0, 3), dtype=torch.long)
    if aff.dim() != 2 or aff.shape[0] != aff.shape[1]:
        raise ValueError(f"aff must be square [K, K], got shape {tuple(aff.shape)}")
    if n_triplets <= 0:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)

    n_clusters = aff.shape[0]
    if n_clusters < 3:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)

    aff_sym = 0.5 * (aff + aff.t())
    aff_sym = aff_sym.clone()
    aff_sym.fill_diagonal_(0.0)

    anchors = torch.randint(0, n_clusters, (n_triplets,), device=aff.device)
    triplets = []

    max_val = aff_sym.max().detach()

    for anchor in anchors:
        pos_prob = aff_sym[anchor].clone()
        pos_prob[anchor] = 0.0
        if pos_prob.sum() <= 0:
            pos_prob.fill_(1.0)
            pos_prob[anchor] = 0.0
        pos = torch.multinomial(pos_prob, 1).squeeze(0)

        neg_score = (max_val - aff_sym[anchor]).clamp_min(0.0)
        neg_score[anchor] = 0.0
        neg_score[pos] = 0.0
        if neg_score.sum() <= 0:
            neg_score.fill_(1.0)
            neg_score[anchor] = 0.0
            neg_score[pos] = 0.0
        neg = torch.multinomial(neg_score, 1).squeeze(0)

        triplets.append(torch.stack([anchor, pos, neg]))

    if not triplets:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)
    return torch.stack(triplets, dim=0)


def _lca_depth_hyp(x, y, manifold):
    """
    Estimate rooted LCA depth with the hyperbolic Gromov product.
    """
    depth = 0.5 * (manifold.dist0(x) + manifold.dist0(y) - manifold.dist(x, y))
    return depth.clamp_min(0.0)


def hierarchy_triplet_loss_hyp(
    proto_h,
    triplets,
    curvature,
    margin=0.05,
    geometry_model="poincare",
    *,
    objective="ranking_ce",
    affinity=None,
):
    """
    Compare the three Gromov-product depths of each sampled triplet.

    ranking_ce preserves the original implementation: cross-entropy with
    (a,p) as target and a margin on (a,n) and (p,n). Its log-sum-exp form is
    related to Sohn (2016), Eq. (3), with adapted scores and pair construction.

    similarity_weighted uses the form of Long and van Noord (2023), Eq. (10),
    retaining our depth proxy and sampling rather than reproducing all sHHC.
    affinity must be the symmetric, zero-diagonal, globally normalized [K,K]
    EMA matrix. Detached weights are multiplied by K*(K-1) to restore an
    off-diagonal mean of one. No margin is used for this objective.
    Both objectives return the mean over sampled triplets.
    """
    if objective not in HIERARCHY_OBJECTIVES:
        raise ValueError(f"Unknown hierarchy objective {objective!r}; expected {HIERARCHY_OBJECTIVES}")
    if triplets.numel() == 0:
        return proto_h.new_zeros(())
    if proto_h.dim() != 2:
        raise ValueError(f"proto_h must be [K, D], got shape {tuple(proto_h.shape)}")
    if triplets.dim() != 2 or triplets.shape[1] != 3:
        raise ValueError(f"triplets must be [T, 3], got shape {tuple(triplets.shape)}")

    geometry = make_hyperbolic_geometry(geometry_model, curvature)

    anchors = triplets[:, 0]
    positives = triplets[:, 1]
    negatives = triplets[:, 2]

    anc = proto_h[anchors]
    pos = proto_h[positives]
    neg = proto_h[negatives]

    if objective == "ranking_ce":
        # Keep the legacy operations, including their order, unchanged.
        s_ap = _lca_depth_hyp(anc, pos, geometry)
        s_an = _lca_depth_hyp(anc, neg, geometry) + margin
        s_pn = _lca_depth_hyp(pos, neg, geometry) + margin
        scores = torch.stack([s_ap, s_an, s_pn], dim=1)
        target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, target)

    n_clusters = proto_h.shape[0]
    if not isinstance(affinity, torch.Tensor) or affinity.shape != (n_clusters, n_clusters):
        raise ValueError("similarity_weighted requires a [K, K] affinity tensor")
    if not torch.is_floating_point(affinity) or not torch.isfinite(affinity).all():
        raise ValueError("affinity must contain finite floating-point values")
    if (affinity < 0).any():
        raise ValueError("affinity must be non-negative")

    affinity = affinity.detach().to(device=proto_h.device, dtype=proto_h.dtype)
    weights = torch.stack(
        [affinity[anchors, positives], affinity[anchors, negatives], affinity[positives, negatives]],
        dim=1,
    ) * (n_clusters * (n_clusters - 1))

    scores = torch.stack([
        _lca_depth_hyp(anc, pos, geometry),
        _lca_depth_hyp(anc, neg, geometry),
        _lca_depth_hyp(pos, neg, geometry),
    ], dim=1)
    return (weights.sum(dim=1) - (F.softmax(scores, dim=1) * weights).sum(dim=1)).mean()

