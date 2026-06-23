"""Element masking for structural completion pretraining.

Randomly masks element features in a HeteroData graph to create a
self-supervised completion task. The GNN must recover the masked
features from the constraint graph context alone.
"""

from __future__ import annotations

import torch

MASK_TOKEN: float = -1.0
"""Value used to replace masked element features."""


def random_mask(
    data: dict,
    mask_ratio: float = 0.6,
    seed: int | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Randomly mask element node features in a HeteroData-like dictionary.

    Observed elements keep their original ``[x1, y1, x2, y2, confidence]``
    features.  Masked elements get all five features set to ``MASK_TOKEN``
    (``-1.0``).  A mask tensor and the original (target) features are
    returned separately so the trainer can apply a loss only on masked
    elements.

    Args:
        data: A HeteroData-like object (or bare dict) that contains
            ``data["element"].x`` of shape ``(N_elem, 5)``.
        mask_ratio: Fraction of element nodes to mask (default 0.6).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tuple of ``(masked_data, mask_info)`` where:
            - ``masked_data`` has the same structure as ``data``, but with
              masked element features replaced by ``MASK_TOKEN``.
            - ``mask_info`` is a dict with:
                ``"mask"``: ``(N_elem,)`` bool tensor (``True`` = masked).
                ``"target"``: ``(N_elem, 5)`` original features.
    """
    # Get element features.
    try:
        x = data["element"].x
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            "Input data must contain 'data[\"element\"].x'"
        ) from exc

    if x.numel() == 0:
        # No elements — nothing to mask.
        mask = torch.zeros(0, dtype=torch.bool)
        return data, {"mask": mask, "target": x}

    N = x.shape[0]
    if seed is not None:
        rng = torch.Generator().manual_seed(seed)
    else:
        rng = None

    # Random mask.
    mask = torch.rand(N, generator=rng) < mask_ratio

    # Save targets (original features).
    target = x.clone()

    # Replace masked elements' features with MASK_TOKEN.
    x_masked = x.clone()
    x_masked[mask] = MASK_TOKEN
    data["element"].x = x_masked

    return data, {"mask": mask, "target": target}


def compute_mask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on masked elements only.

    Args:
        prediction: ``(N_elem, 5)`` predicted features from completion head.
        target: ``(N_elem, 5)`` original (unmasked) features.
        mask: ``(N_elem,)`` bool tensor (``True`` = compute loss for this element).

    Returns:
        Scalar MSE loss, evaluated only on masked positions.
        Returns 0.0 if no elements are masked.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=prediction.device)
    return torch.nn.functional.mse_loss(
        prediction[mask], target[mask]
    )
