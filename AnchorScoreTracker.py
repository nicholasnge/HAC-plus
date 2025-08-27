import torch

class AnchorScoreTracker:
    """
    Aggregate per-gaussian scores into per-anchor scores over time.
    Call update(gauss_scores, gauss_to_anchor, num_anchors=..., weights=...) each frame.
    Then call get_scores() to retrieve per-anchor averages.
    """
    def __init__(self, device="cuda", dtype=torch.float32, eps=1e-8):
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self._sum = None          # [N_anchors]
        self._cnt = None          # [N_anchors]
        self._num_anchors = 0

    def reset(self, num_anchors: int):
        self._num_anchors = int(num_anchors)
        self._sum = torch.zeros(self._num_anchors, device=self.device, dtype=self.dtype)
        self._cnt = torch.zeros(self._num_anchors, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(self,
               gauss_scores: torch.Tensor,
               gauss_to_anchor: torch.Tensor,
               num_anchors: int):
        """
        gauss_scores    : [G]  per-gaussian scores (after rasterization)
        gauss_to_anchor : [G]  int64 indices mapping each gaussian -> anchor id in the ORIGINAL anchor array
        num_anchors     : int  (recommended) current number of anchors to track
        """
        if gauss_scores is None or gauss_scores.numel() == 0:
            return

        # (Re)initialize buffers on first call or if N changes
        if (self._sum is None) or (num_anchors != self._num_anchors):
            print("RESETTING ANCHOR SCORE TRACKER")
            self.reset(num_anchors)

        # Sanitize inputs
        s = gauss_scores.detach().to(self.dtype).to(self.device).view(-1)
        a = gauss_to_anchor.detach().to(torch.long).to(self.device).view(-1)

        # Scatter-add into per-anchor accumulators
        self._sum.index_add_(0, a, s)
        ones = torch.ones_like(s, dtype=self._cnt.dtype, device=self.device)
        self._cnt.index_add_(0, a, ones)
        

    @torch.no_grad()
    def get_scores(self) -> torch.Tensor:
        """
        Returns per-anchor average scores, shape [N_anchors].
        Anchors never hit will have 0 (due to eps in denominator).
        """
        if self._sum is None:
            return None
        return self._sum / (self._cnt + self.eps)

    @torch.no_grad()
    def get_counts(self) -> torch.Tensor:
        """Returns per-anchor total weight accumulated (useful for debugging/thresholds)."""
        return self._cnt.clone() if self._cnt is not None else None
