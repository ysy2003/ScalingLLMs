"""
Structural Alignment metrics (@ysy)

This module computes structural alignment between a reference DOM and a
predicted/rendered DOM, including:
- Tree Edit Similarity (derived from tree edit distance)
- Semantic HTML Usage ratio
- Accessibility Score (alt coverage + ARIA coverage)
"""

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class StructuralAlignmentScores:
    """Scores for structural alignment between reference and prediction."""
    tree_edit_similarity: float      # [0, 1]
    semantic_html_ratio: float       # [0, 1]
    accessibility_score: float       # [0, 1]


# Common semantic tags used to encourage meaningful HTML structure
SEMANTIC_TAGS = {
    "header", "nav", "main", "section", "article", "aside", "footer"
}


def tree_edit_distance(ref_dom: Any, pred_dom: Any) -> int:
    """
    A simple, heuristic tree edit distance:

    - Cost 1 for:
        * deleting an entire subtree
        * inserting an entire subtree
        * replacing a node whose tag differs
    - Children are aligned by index (no reordering).

    This is NOT the full Zhang–Shasha algorithm, but is often
    good enough as a structural similarity signal for DOM trees.
    """
    # both None → no cost
    if ref_dom is None and pred_dom is None:
        return 0

    # one None → cost = size of the other tree
    if ref_dom is None:
        return _count_nodes(pred_dom)
    if pred_dom is None:
        return _count_nodes(ref_dom)

    # cost on the root
    tag_a = getattr(ref_dom, "tag", "").lower()
    tag_b = getattr(pred_dom, "tag", "").lower()
    cost = 0 if tag_a == tag_b else 1

    # children
    children_a = list(getattr(ref_dom, "children", []))
    children_b = list(getattr(pred_dom, "children", []))

    len_a = len(children_a)
    len_b = len(children_b)
    common = min(len_a, len_b)

    # pairwise distance for aligned children
    for i in range(common):
        cost += tree_edit_distance(children_a[i], children_b[i])

    # remaining children in A → deletions
    for i in range(common, len_a):
        cost += _count_nodes(children_a[i])

    # remaining children in B → insertions
    for i in range(common, len_b):
        cost += _count_nodes(children_b[i])

    return cost



def _count_nodes(node: Any) -> int:
    """Total number of nodes in a DOM tree."""
    if node is None:
        return 0
    total = 1
    for child in getattr(node, "children", []):
        total += _count_nodes(child)
    return total


def tree_edit_similarity(ref_dom: Any, pred_dom: Any) -> float:
    """
    Similarity score in [0, 1] derived from tree edit distance.
    1.0 = identical trees, 0.0 = completely different.
    """
    if ref_dom is None and pred_dom is None:
        return 1.0

    distance = tree_edit_distance(ref_dom, pred_dom)
    max_nodes = max(_count_nodes(ref_dom), _count_nodes(pred_dom))
    if max_nodes == 0:
        return 1.0
    # normalize: similarity = 1 - (distance / max_nodes)
    return max(0.0, 1.0 - distance / max_nodes)


def semantic_html_usage(root: Any) -> float:
    """
    Ratio of semantic tags to semantic + generic <div> tags.
    Returns 0.0 if neither appears.
    """
    semantic_count, div_count = _count_semantic_and_div(root)
    denom = semantic_count + div_count
    if denom == 0:
        return 0.0
    return semantic_count / denom


def _count_semantic_and_div(node: Any) -> Tuple[int, int]:
    """Traverse DOM tree to count semantic tags and <div> tags."""
    if node is None:
        return 0, 0

    semantic_count = 0
    div_count = 0

    tag = getattr(node, "tag", "").lower()
    if tag in SEMANTIC_TAGS:
        semantic_count += 1
    elif tag == "div":
        div_count += 1

    for child in getattr(node, "children", []):
        sc, dc = _count_semantic_and_div(child)
        semantic_count += sc
        div_count += dc

    return semantic_count, div_count


def accessibility_score(root: Any) -> float:
    """
    Composite accessibility score based on:
      - fraction of <img> with non-empty alt
      - fraction of elements with ARIA roles / labels

    Returns a score in [0, 1]. Current definition:
        score = 0.5 * alt_coverage + 0.5 * aria_coverage
    You can adjust weights later if needed.
    """
    total_imgs, imgs_with_alt = _count_images_with_alt(root)
    total_aria_candidates, nodes_with_aria = _count_nodes_with_aria(root)

    alt_coverage = imgs_with_alt / total_imgs if total_imgs > 0 else 0.0
    aria_coverage = (
        nodes_with_aria / total_aria_candidates
        if total_aria_candidates > 0 else 0.0
    )

    return 0.5 * alt_coverage + 0.5 * aria_coverage


def _count_images_with_alt(node: Any) -> Tuple[int, int]:
    if node is None:
        return 0, 0

    total_imgs = 0
    imgs_with_alt = 0

    tag = getattr(node, "tag", "").lower()
    attrs = getattr(node, "attrs", {})  # assume dict-like

    if tag == "img":
        total_imgs += 1
        alt = attrs.get("alt", "")
        if isinstance(alt, str) and alt.strip():
            imgs_with_alt += 1

    for child in getattr(node, "children", []):
        ti, ia = _count_images_with_alt(child)
        total_imgs += ti
        imgs_with_alt += ia

    return total_imgs, imgs_with_alt


def _count_nodes_with_aria(node: Any) -> Tuple[int, int]:
    if node is None:
        return 0, 0

    total = 0
    with_aria = 0

    attrs = getattr(node, "attrs", {})
    # any node can in principle carry ARIA attributes
    total += 1
    if any(
        k.startswith("aria-") for k in attrs.keys()
    ) or "role" in attrs or "aria-label" in attrs:
        with_aria += 1

    for child in getattr(node, "children", []):
        t, w = _count_nodes_with_aria(child)
        total += t
        with_aria += w

    return total, with_aria


def compute_structural_alignment_scores(
    ref_dom: Any,
    pred_dom: Any,
) -> StructuralAlignmentScores:
    """
    High-level entry point for Structural Alignment.
    """
    tes = tree_edit_similarity(ref_dom, pred_dom)
    semantic_ratio = semantic_html_usage(pred_dom)
    acc_score = accessibility_score(pred_dom)

    return StructuralAlignmentScores(
        tree_edit_similarity=tes,
        semantic_html_ratio=semantic_ratio,
        accessibility_score=acc_score,
    )


def compute_overall_structural_alignment(
    ref_dom: Any,
    pred_dom: Any,
) -> float:
    """
    Optional helper if you want a single scalar for Structural Alignment,
    e.g., a weighted combination of:
      - tree_edit_similarity
      - semantic_html_ratio
      - accessibility_score
    """
    scores = compute_structural_alignment_scores(ref_dom, pred_dom)
    # default: simple average; adjust weights as needed
    return (
        scores.tree_edit_similarity
        + scores.semantic_html_ratio
        + scores.accessibility_score
    ) / 3.0


__all__ = [
    "StructuralAlignmentScores",
    "SEMANTIC_TAGS",
    "tree_edit_distance",
    "tree_edit_similarity",
    "semantic_html_usage",
    "accessibility_score",
    "compute_structural_alignment_scores",
    "compute_overall_structural_alignment",
]
