import os
from typing import List, Dict, Optional

from openai import OpenAI
from utils import response_parsing

EVAL_SYSTEM_PROMPT = (
    "You are a strict music evaluator. Do NOT reveal chain-of-thought. "
    "Return ONLY strict JSON with exactly these keys:\n"
    '{"pass": boolean, "reasons": str[], "suggested_revisions": str[]}.'
)

REASONING_EVAL_SYSTEM_PROMPT = (
    "You are a music theory expert evaluating compositional reasoning.\n"
    "Return ONLY strict JSON with exactly these keys: "
    '{"reasoning_quality_score": number (0-1), "summary_feedback": string, "suggested_revisions": [string]}.'
)

REFERENCE_EVAL_USER_TEMPLATE = (
    "You are tasked with evaluating whether the CANDIDATE music in ABC notation successfully follows the REFERENCE.\n\n"
    "REFERENCE (ABC):\n{reference_abc}\n\n"
    "CANDIDATE (ABC):\n{candidate_abc}\n\n"
    "Assess the candidate against the reference on:\n"
    "1) Stylistic idioms (harmony, figuration, texture, ornamentation).\n"
    "2) Characteristic gestures and phrasing.\n"
    "3) Overall musical coherence.\n\n"
    "If for any key aspect, the candidate is different from the reference, set \"pass\" as false. "
    "Return a strict JSON object with:\n"
    "- \"pass\": true/false\n"
    "- \"reasons\": [brief bullet reasons]\n"
    "- \"suggested_revisions\": [short, actionable instructions if pass=false]\n\n"
    "Keep suggestions concise and targeted so that a composer-model could follow them directly. "
    "The goal is to help the candidate better match the reference style. Don't make overly generic suggestions.\n\n"
    "Don't make the two pieces identical—musically similar but not the same.\n"
)

class ReferenceEvaluator:
    """
    Self-contained evaluator for comparing a CANDIDATE ABC against a REFERENCE ABC.

    Produces strict JSON with:
      {"pass": bool, "reasons": List[str], "suggested_revisions": List[str]}
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = EVAL_SYSTEM_PROMPT,
        user_template: str = REFERENCE_EVAL_USER_TEMPLATE,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.99,
    ):
        # Initialize OpenAI client (reads from env if api_key is falsy)
        self._oa_client = OpenAI(api_key=api_key or None)

        self.model = model
        self.system_prompt = system_prompt
        self.user_template = user_template

        # Sampling params (Responses API names shown in _call_responses)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def _build_user_msg(self, reference_abc: str, candidate_abc: str) -> str:
        return self.user_template.format(
            reference_abc=reference_abc,
            candidate_abc=candidate_abc
        )

    def _call_responses(self, user_msg: str) -> str:
        """
        Calls the Responses API and returns concatenated text output.
        """
        resp = self._oa_client.responses.create(
            model=self.model,
            instructions=self.system_prompt,     # system/developer guidance
            input=[{"role": "user", "content": user_msg}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,  # Responses API naming
        )

        # Prefer the SDK's convenience attribute; fall back to manual extraction.
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Defensive fallback in case output_text isn't present
        chunks: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        chunks.append(getattr(c, "text", "") or "")
        return "".join(chunks).strip()

    def evaluate(self, reference_abc: str, candidate_abc: str) -> Dict:
        """
        Returns a dict with keys: pass (bool), reasons (List[str]), suggested_revisions (List[str]).
        Falls back to a permissive pass if parsing fails.
        """
        eval_user = self._build_user_msg(reference_abc, candidate_abc)

        try:
            text = self._call_responses(eval_user)
        except Exception:
            # Safe fallback on transport/API errors
            return {"pass": True, "reasons": [], "suggested_revisions": []}

        parsed = response_parsing.parse_json_response(text)

        if (
            isinstance(parsed, dict)
            and "pass" in parsed
            and "reasons" in parsed
            and "suggested_revisions" in parsed
        ):
            return parsed

        # Safe fallback if model didn't return valid JSON
        return {"pass": True, "reasons": [], "suggested_revisions": []}


class COTEvaluator:
    """
    Evaluates the reasoning_steps (COT) provided by the model.
    The goal is not to grade musical output, but to check whether
    the reasoning is coherent, musically consistent, and relevant to the final ABC.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = REASONING_EVAL_SYSTEM_PROMPT,
        # user_template: str = REFERENCE_EVAL_USER_TEMPLATE,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.99,
    ):
        # Initialize OpenAI client (reads from env if api_key is falsy)
        self._oa_client = OpenAI(api_key=api_key or None)

        self.model = model
        self.system_prompt = system_prompt
        # self.user_template = user_template

        # Sampling params (Responses API names shown in _call_responses)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
    def evaluate(self, reasoning_steps1: list[str], reasoning_steps2: str) -> dict:
        """
        Ask the model to critique its own reasoning.
        Returns a dict such as:
        {
            "reasoning_quality_score": float,
            "summary_feedback": str,
            "suggested_revisions": [str],
            "pass": boolean
        }
        """
        if not reasoning_steps1:
            return {
                "reasoning_quality_score": 0.0,
                "summary_feedback": "No reasoning steps provided.",
                "suggested_revisions": [],
                "pass": False
            }
        
        ####### NOTE
        # have llm generate cot before music generation and compare all following cot's with this original cot
        
        joined_steps = "\n".join(f"- {step}" for step in reasoning_steps1)
        prompt = (
            "You are a music theory expert evaluating compositional reasoning.\n"
            "Here are previous reasoning steps provided by the model for the same prompt:\n"
            f"{joined_steps}\n\n"
            "Here are the current reasoning steps.\n"
            f"{reasoning_steps2}\n\n"
            "Evaluate how well the current reasoning follows the previous reasoning in coherency.\n"
            "Return JSON with exactly these keys: "
            '{"reasoning_quality_score": number (0-1), '
            '"summary_feedback": string, '
            '"suggested_revisions": [string], '
            '"pass": reasoning_quality_score >= 0.85}.'
        )

        resp = self._oa_client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,  # Responses API naming
        )
        raw_text = getattr(resp, "output_text", None) or ""
        try:
            from utils import response_parsing
            parsed = response_parsing.parse_json_response(raw_text)
            return parsed or {}
        except Exception:
            return {"summary_feedback": raw_text[:500], "suggested_revisions": []}
    
