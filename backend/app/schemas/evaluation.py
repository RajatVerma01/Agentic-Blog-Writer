from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from app.config.settings import get_settings as _get_settings

_settings = _get_settings()
APPROVAL_THRESHOLD: float = _settings.EVALUATION_THRESHOLD
MAX_REVISION_CYCLES: int = _settings.MAX_REVISION_CYCLES



class RubricDimension(str, Enum):
    """
    The five evaluation dimensions.
    Using an Enum ensures agents and the API always use the same dimension names.
    """
    GRAMMAR_CLARITY   = "grammar_clarity"
    FACTUAL_ACCURACY  = "factual_accuracy"
    CITATION_QUALITY  = "citation_quality"
    STRUCTURE_FLOW    = "structure_flow"
    SEO_OPTIMIZATION  = "seo_optimization"


RUBRIC_WEIGHTS: dict[RubricDimension, float] = {
    RubricDimension.GRAMMAR_CLARITY:  0.25,
    RubricDimension.FACTUAL_ACCURACY: 0.25,
    RubricDimension.CITATION_QUALITY: 0.20,
    RubricDimension.STRUCTURE_FLOW:   0.15,
    RubricDimension.SEO_OPTIMIZATION: 0.15,
}




class ImprovementPoint(BaseModel):
    """
    A single, specific, actionable feedback item for the Writer agent.

    Design principle: Vague feedback ("add more citations") is useless.
    Every improvement point must name EXACTLY what is wrong and EXACTLY
    what to write instead.

    Example:
        {
            "dimension":  "citation_quality",
            "issue":      "The claim '94% diagnostic accuracy' has no source cited.",
            "suggestion": "Add citation from the Stanford AI Health Lab 2023 study."
        }
    """

    dimension: RubricDimension = Field(
        description="Which rubric dimension this improvement addresses.",
        examples=["citation_quality", "factual_accuracy"],
    )

    issue: str = Field(
        min_length=10,
        max_length=500,
        description="A specific description of what is wrong in the current draft.",
        examples=["The claim '94% accuracy' in Section 2 has no source cited."],
    )

    suggestion: str = Field(
        min_length=10,
        max_length=500,
        description="A concrete, actionable instruction for how to fix the issue.",
        examples=["Add: (Source: Stanford AI Health Lab, 2023) after the claim."],
    )

    @field_validator("issue", "suggestion", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class DimensionScore(BaseModel):
    """
    Score for a single rubric dimension, with optional rationale.

    Example:
        {
            "dimension": "citation_quality",
            "score":     4.5,
            "rationale": "3 out of 7 claims have no source URL."
        }
    """

    dimension: RubricDimension

    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Score for this dimension on a scale of 0.0 to 10.0.",
        examples=[7.5, 4.0],
    )

    rationale: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Brief explanation of why this score was given.",
        examples=["Good grammar throughout but two run-on sentences in Section 3."],
    )

    @property
    def weight(self) -> float:
        """Returns the rubric weight for this dimension."""
        return RUBRIC_WEIGHTS[self.dimension]

    @property
    def weighted_contribution(self) -> float:
        """Returns this dimension's contribution to the final score."""
        return round(self.score * self.weight, 4)


class EvaluationResult(BaseModel):
    """
    The full evaluation report produced by the Evaluator agent after
    reviewing one draft from the Writer agent.

    This object is:
    - Written to BlogState["evaluation_result"] by the evaluator node.
    - Read by the writer node (feedback + improvements) for revisions.
    - Embedded in BlogResult and returned to the user via the API.

    Computed fields (score, approved) are derived automatically from
    dimension_scores, so the Evaluator agent only needs to provide
    per-dimension scores.

    JSON shape expected from Evaluator LLM output:
    {
        "dimension_scores": [
            {"dimension": "grammar_clarity",  "score": 8.0, "rationale": "..."},
            {"dimension": "factual_accuracy",  "score": 5.5, "rationale": "..."},
            {"dimension": "citation_quality",  "score": 4.0, "rationale": "..."},
            {"dimension": "structure_flow",    "score": 7.5, "rationale": "..."},
            {"dimension": "seo_optimization",  "score": 6.0, "rationale": "..."}
        ],
        "improvements": [
            {
                "dimension":  "citation_quality",
                "issue":      "Claim about 94% accuracy has no source.",
                "suggestion": "Add the Stanford 2023 study citation."
            }
        ],
        "feedback": "Good grammar and structure. Main issues are missing citations..."
    }
    """

    dimension_scores: list[DimensionScore] = Field(
        min_length=5,
        max_length=5,
        description=(
            "Scores for all five rubric dimensions. "
            "Must contain exactly one entry per dimension."
        ),
    )

    improvements: list[ImprovementPoint] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "List of specific, actionable improvement points for the Writer. "
            "Empty list means the blog passed all checks."
        ),
    )

    feedback: str = Field(
        min_length=20,
        max_length=1000,
        description=(
            "A 2-4 sentence summary paragraph of the evaluation. "
            "Written to the Writer agent — constructive and specific."
        ),
        examples=[
            "Good grammar and logical structure. The main weakness is citation quality — "
            "3 claims lack sources. Fix the factual error in Section 2 about AI replacing doctors."
        ],
    )

    revision_number: int = Field(
        default=0,
        ge=0,
        description="Which revision cycle this evaluation is for (0 = first draft).",
    )

    @field_validator("dimension_scores")
    @classmethod
    def validate_all_dimensions_present(
        cls, scores: list[DimensionScore]
    ) -> list[DimensionScore]:
        """
        Ensures all 5 rubric dimensions are present exactly once.
        Prevents the Evaluator LLM from omitting a dimension silently.
        """
        provided = {s.dimension for s in scores}
        required = set(RubricDimension)
        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing dimension scores for: {[d.value for d in missing]}. "
                f"All 5 dimensions are required."
            )
        duplicates = [
            d for d in provided
            if sum(1 for s in scores if s.dimension == d) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate dimension scores detected: {duplicates}"
            )
        return scores


    @property
    def score(self) -> float:
        """
        Weighted average score across all 5 dimensions.
        Computed automatically from dimension_scores × RUBRIC_WEIGHTS.

        Example:
            grammar_clarity:  8.0 × 0.25 = 2.000
            factual_accuracy: 5.5 × 0.25 = 1.375
            citation_quality: 4.0 × 0.20 = 0.800
            structure_flow:   7.5 × 0.15 = 1.125
            seo_optimization: 6.0 × 0.15 = 0.900
            ─────────────────────────────────────
            TOTAL SCORE               = 6.200
        """
        total = sum(s.weighted_contribution for s in self.dimension_scores)
        return round(total, 2)

    @property
    def approved(self) -> bool:
        """
        True if the weighted score meets the approval threshold (7.0).
        The graph.py conditional edge reads this to decide:
          approved=True  → END (return final blog to user)
          approved=False → back to Writer for revision
        """
        return self.score >= APPROVAL_THRESHOLD

    @property
    def scores_by_dimension(self) -> dict[str, float]:
        """
        Returns a flat dict of dimension → score for easy serialization.
        Example: {"grammar_clarity": 8.0, "citation_quality": 4.0, ...}
        """
        return {s.dimension.value: s.score for s in self.dimension_scores}

    @property
    def weakest_dimension(self) -> DimensionScore:
        """Returns the DimensionScore with the lowest score."""
        return min(self.dimension_scores, key=lambda s: s.score)

    @property
    def strongest_dimension(self) -> DimensionScore:
        """Returns the DimensionScore with the highest score."""
        return max(self.dimension_scores, key=lambda s: s.score)

    def to_writer_prompt_context(self) -> str:
        """
        Formats the evaluation result as a concise text block to inject
        into the Writer agent's revision prompt.

        The Writer reads this to understand exactly what to fix.

        Returns:
            A formatted string like:
            ---
            EVALUATION FEEDBACK (Revision 1)
            Overall Score: 6.2 / 10 — Needs improvement

            Dimension Scores:
              grammar_clarity:  8.0  (weight: 25%)
              factual_accuracy: 5.5  (weight: 25%) ← WEAKEST
              ...

            Specific Issues to Fix:
              [citation_quality] The claim '94% accuracy' has no source.
              → Fix: Add the Stanford 2023 study citation.
            ---
        """
        lines = [
            f"EVALUATION FEEDBACK (Revision {self.revision_number})",
            f"Overall Score: {self.score} / 10 — "
            + ("APPROVED ✓" if self.approved else "Needs improvement"),
            "",
            "Dimension Scores:",
        ]

        weakest = self.weakest_dimension  # compute once, not on every iteration
        for dim_score in sorted(
            self.dimension_scores, key=lambda s: s.score
        ):
            weight_pct = int(RUBRIC_WEIGHTS[dim_score.dimension] * 100)
            marker = " ← WEAKEST" if dim_score == weakest else ""
            lines.append(
                f"  {dim_score.dimension.value:<22} {dim_score.score:>4.1f}  "
                f"(weight: {weight_pct}%){marker}"
            )
            if dim_score.rationale:
                lines.append(f"    Rationale: {dim_score.rationale}")

        if self.improvements:
            lines.append("")
            lines.append("Specific Issues to Fix (address ALL of these):")
            for i, point in enumerate(self.improvements, 1):
                lines.append(
                    f"  {i}. [{point.dimension.value}] {point.issue}"
                )
                lines.append(f"     → Fix: {point.suggestion}")

        lines.append("")
        lines.append(f"Summary: {self.feedback}")

        return "\n".join(lines)


def evaluation_to_state_dict(result: EvaluationResult) -> dict:
    """
    Converts an EvaluationResult to a plain dict for storage in BlogState.
    Includes computed properties (score, approved) that are not stored
    as model fields.

    Used by the evaluator agent node to write to state:
        state["evaluation_result"] = evaluation_to_state_dict(result)
    """
    data = result.model_dump()
    data["score"] = result.score
    data["approved"] = result.approved
    data["scores_by_dimension"] = result.scores_by_dimension
    return data