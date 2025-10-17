import os
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Tuple, List, Any, Dict
from openai import OpenAI

from pdmx import load as pdmx_load, write_musicxml as pdmx_write_musicxml  # noqa: F401
from utils import response_parsing
from rag import RAG
from evaluation import ReferenceEvaluator, COTEvaluator

API_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1")

DEFAULT_USER_PROMPT = "Write a Bach-style Chorale in ABC notation."
# DEFAULT_USER_PROMPT = "Write a Chopin-style Nocturne in ABC notation."
# DEFAULT_USER_PROMPT = "Write Mozart-style string quartet in ABC notation."


def _write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(path)


class VComposerClient:
    def __init__(self, api_key: str, model: str, out_dir: str):
        # Initialize OpenAI client
        self._oa_client = OpenAI(api_key=api_key or None)

        self.model = model
        self.system_prompt = (
            "You are a helpful assistant and an excellent composer. "
            "Do NOT reveal chain-of-thought or step-by-step reasoning in the final answer. "
            "Return ONLY strict JSON with exactly these keys: "
            '{"answer": string, "detailed steps": string[]}. No extra text, no code fences.'
        )
        self.system_prompt_reasoning = (
            "You are a helpful assistant and an excellent composer. "
            "Return ONLY strict JSON with exactly these keys: "
            '{"answer": string, "reasoning_steps": string[]}. '
            "The 'answer' must be an original piece in valid ABC notation only. "
            "The 'reasoning_steps' must be a short ordered list of concise statements describing the compositional choices you will follow and their reasonings. "
            "Do NOT include extra text or code fences."
        )
        
        self.system_prompt_reasoning_only = (
            "You are a helpful assistant and an expert in musical reasoning."
            "Return ONLY strict JSON with exactly these keys: "
            '{"reasoning_steps": string[]}. '
            "The 'reasoning_steps' must be a short ordered list of concise statements describing the compositional choices you will follow and their reasonings. "
            "Do NOT include extra text or code fences."
        )

        self.rag = RAG('/data/PDMX', api_key=api_key, model=model, output_dir=out_dir)
        self.reference_evaluator = ReferenceEvaluator(api_key=api_key, model=self.model)
        self.cot_evaluator = COTEvaluator(api_key=api_key, model=model)

        self.last_reference_abc: Optional[str] = None
        self.out_dir = out_dir

    def _compose_once(self, user_prompt: str) -> Tuple[Dict[str, Any], str]:
        """
        Single composition call.
        Returns (parsed_json_obj, raw_text).
        """
        resp = self._oa_client.responses.create(
            model=self.model,
            # In the Responses API, use `instructions` for system/developer guidance.
            instructions=self.system_prompt,
            input=[{"role": "user", "content": user_prompt}],
        )
        # Helper to extract text
        raw_text = getattr(resp, "output_text", None) or self._extract_output_text(resp)
        obj = response_parsing.parse_json_response(raw_text or "")
        return obj, (raw_text or "")
    
    ## new method with reasoning
    def _compose_once_with_reasoning(self, user_prompt: str) -> Tuple[Dict[str, Any], str]:
        """
        Single call that returns both the composition (answer) and a structured reasoning trace.
        Returns (parsed_json_obj, raw_text). Parsed JSON expected to have:
          {"answer": "<ABC>", "reasoning_steps": ["step1", "step2", ...]}
        """
        
        resp = self._oa_client.responses.create(
            model=self.model,
            temperature=0.7,
            instructions=self.system_prompt_reasoning,
            input=[{"role": "user", "content": user_prompt}],
        )
        
        # extract text
        raw_text = getattr(resp, "output_text", None) or self._extract_output_text(resp)
        obj = response_parsing.parse_json_response(raw_text or "")
        return obj, (raw_text or "")
    
    
    ## new method only reasoning
    def _only_reasoning(self, user_prompt: str) -> Tuple[Dict[str, Any], str]:
        """
        Single call for reasoning at the beginning of loop.
        Returns (parsed_json_obj, raw_text).
        """
        resp = self._oa_client.responses.create(
            model=self.model,
            # In the Responses API, use `instructions` for system/developer guidance.
            instructions=self.system_prompt_reasoning_only,
            input=[{"role": "user", "content": user_prompt}],
        )
        # Helper to extract text
        raw_text = getattr(resp, "output_text", None) or self._extract_output_text(resp)
        obj = response_parsing.parse_json_response(raw_text or "")
        return obj, (raw_text or "") 
    

    @staticmethod
    def _extract_output_text(resp) -> str:
        """
        Fallback extractor in case `.output_text` is unavailable on the returned object.
        Tries to concatenate text from the output array.
        """
        try:
            # SDK usually has resp.output_text; this is a defensive fallback.
            chunks = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            chunks.append(getattr(c, "text", "") or "")
            return "".join(chunks)
        except Exception:
            return ""

    def ask_llm(self, user_prompt: str, reference_eval_agent: bool, agent_max_rounds: int = 3):
        """Compose music; optionally self-evaluate against a RAG reference and refine."""
        # --- RAG & initial prompt ---
        prompt, reference_abc = self.apply_rag(user_prompt)
        print("Final user prompt sent to LLM. Composing...")

        # Fast path (no refinement loop or no reference to compare against)
        # if not reference_eval_agent or not reference_abc:
        #     obj, raw = self._compose_once(prompt)
        #     return self.extract_abc(obj, raw), obj, raw
        
        if not reference_eval_agent or not reference_abc:
            if getattr(self, "use_cot_generation", False):
                print("no but cot")
                obj, raw = self._compose_once_with_reasoning(prompt)
            else:
                obj, raw = self._compose_once(prompt)
            return self.extract_abc(obj, raw), obj, raw
        
        ## New: Get initial reasoning for COT
        if getattr(self, "use_cot_generation", False):
            last_obj_reasoning, last_raw_reasoning = self._only_reasoning(DEFAULT_USER_PROMPT)
            initial_reasoning = last_obj_reasoning.get("reasoning_steps") if isinstance(last_obj_reasoning, dict) else None
            print("INITIAL REASONING:", initial_reasoning)

        # --- Refinement loop (evaluate -> suggest -> regenerate) ---
        all_revisions_history: List[List[str]] = []
        all_evals: List[Dict[str, Any]] = []
        last_obj: Dict[str, Any] = {}
        last_raw: str = ""
        last_abc: str = ""
        rounds = max(1, agent_max_rounds)

        for i in range(rounds):
            # Compose
            # last_obj, last_raw = self._compose_once(prompt)
            # last_abc = self.extract_abc(last_obj, last_raw)
            # round_abc = os.path.join(self.out_dir, f'round_{i}_result.abc')
            # self.save_abc(last_abc, round_abc)
            
            #### NEW
            # Compose: use COT generation if enabled
            if getattr(self, "use_cot_generation", False):
                last_obj, last_raw = self._compose_once_with_reasoning(prompt)
                # print("OBJ, RAW", last_obj, last_raw)
                print("doing cot!! :)")
            else:
                last_obj, last_raw = self._compose_once(prompt)
                # print("OBJ, RAW", last_obj, last_raw)

            last_abc = self.extract_abc(last_obj, last_raw)
            round_abc = os.path.join(self.out_dir, f'round_{i}_result.abc')
            self.save_abc(last_abc, round_abc)

            # If the model returned reasoning, save it alongside the round for later analysis
            try:
                reasoning_steps = last_obj.get("reasoning_steps") if isinstance(last_obj, dict) else None
                print("FULL REASONINGGGG:", reasoning_steps)
                if reasoning_steps:
                    reason_path = os.path.join(self.out_dir, f'round_{i}_reasoning.json')
                    _write_text(reason_path, json.dumps({"reasoning_steps": reasoning_steps}, indent=2))
            except Exception:
                # be robust: don't fail the pipeline if saving reasoning fails
                pass
            
            # --- Evaluate reasoning trace (if available) ---
            cot_ev = {}
            ev = {}
            if getattr(self, "use_cot_generation", False):
                print("using cot evaluator!!!")
                reasoning_steps = (
                    last_obj.get("reasoning_steps")
                    if isinstance(last_obj, dict)
                    else None
                )
                cot_ev = self.cot_evaluator.evaluate(initial_reasoning, reasoning_steps)
                ev["cot_evaluation"] = cot_ev  # attach reasoning eval to the same eval dict
            else:
                # Evaluate
                print("\nEvaluating the composition against the reference...")
                ev = self.reference_evaluator.evaluate(reference_abc, last_abc) or {}
    
            all_evals.append({"round": i, "evaluation": ev})

            # ABC -> MusicXML
            subprocess.run(
                ['python', 'utils/abc2xml.py', '-o', self.out_dir, round_abc],
                check=False
            )
            
            if getattr(self, "use_cot_generation", False):
                print("using cot pass!!")
                cot_score = cot_ev.get("reasoning_quality_score", 0.0)
                cot_pass = cot_ev.get("pass", cot_score >= 0.85)
                print(f"[COT] Pass decision based on reasoning quality ({cot_score:.2f}): {cot_pass}")
                if cot_pass:
                    print("COT evaluation passed. Using current composition.")
                    break
                
            else:
                if ev.get("pass", False):
                    print("Evaluation passed. Using current composition.")
                    break

            # Current-round suggestions only
            current_revisions = [
                s for s in (ev.get("suggested_revisions") or [])
                if isinstance(s, str) and s.strip()
            ]
            if not current_revisions:
                current_revisions = self.default_revisions()

            all_revisions_history.append(current_revisions)
            print(f"Round {i} revisions:", current_revisions)

            # Stop if budget reached; otherwise rebuild prompt fresh with only current revisions
            if i == rounds - 1:
                print(f"Reached refinement budget ({agent_max_rounds}). Proceeding with latest result.")
                break

            #### NEW ADDED COT
            if cot_ev.get("suggested_revisions"):
                current_revisions += cot_ev["suggested_revisions"]
    
            prompt = self.build_prompt(user_prompt, reference_abc, current_revisions)
            print("Regenerating with current round's revision instructions...")

        # Write a concise summary (Markdown)
        self._write_summary_report(
            user_prompt=user_prompt,
            reference_present=bool(reference_abc),
            all_evals=all_evals,
            all_revisions_history=all_revisions_history,
            final_round_abc=last_abc
        )

        return last_abc, last_obj, last_raw

    def extract_abc(self, obj: Dict[str, Any], raw: str) -> str:
        cand = obj.get("answer", raw) if isinstance(obj, dict) else raw
        abc = response_parsing.extract_abc(cand) or (
            None if cand is raw else response_parsing.extract_abc(raw)
        )
        return (abc or cand).strip()

    def build_prompt(self, base: str, ref_abc: str, revisions: Optional[List[str]] = None) -> str:
        head = f"{base}\n\nUse the following as reference (ABC):\n{ref_abc}\n\n"
        if revisions:
            bullets = "\n".join(f"- {r}" for r in revisions if r.strip())
            return (
                f"{head}"
                "Revise to better follow the reference using these instructions:\n"
                f"{bullets}\n"
                "Follow the style and approximate phrase/length structure.\n"
                "Return valid ABC notation."
            )
        return (
            f"{head}"
            "Follow the style and approximate length/phrase structure.\n"
            "Return valid ABC notation."
        )

    def default_revisions(self) -> List[str]:
        return [
            "Match the reference's phrase count and approximate length",
            "Increase characteristic stylistic gestures of the reference.",
        ]

    def save_abc(self, abc_text: str, out_path: str) -> str:
        # Normalize line endings and ensure trailing newline
        abc_text = abc_text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(abc_text)
        return os.path.abspath(out_path)

    def _write_summary_report(
        self,
        user_prompt: str,
        reference_present: bool,
        all_evals: List[Dict[str, Any]],
        all_revisions_history: List[List[str]],
        final_round_abc: str,
    ) -> None:
        # Determine first pass round (if any)
        pass_round = None
        for item in all_evals:
            ev = item.get("evaluation") or {}
            cot_ev = ev.get("cot_evaluation", {})
            # if ev.get("pass", False):
            #     pass_round = item["round"]
            #     break
            
            ####### NEW
            pass_flag = (
                cot_ev.get("pass", False)
                if getattr(self, "use_cot_generation", False)
                else ev.get("pass", False)
            )
            if pass_flag:
                pass_round = item["round"]
                break

        # Markdown summary
        md_lines = [
            "# Composition Summary Report",
            "",
            f"- **Reference used:** {'Yes' if reference_present else 'No'}",
            f"- **Total rounds:** {len(all_evals)}",
            f"- **Pass round:** {pass_round if pass_round is not None else 'None'}",
            "",
            "## Per-Round Revisions and Evaluations",
        ]

        for i, item in enumerate(all_evals):
            rnd = item["round"]
            ev = item.get("evaluation") or {}
            cot_ev = ev.get("cot_evaluation", {})
            md_lines.append(f"### Round {rnd}")
            # md_lines.append(f"- **Pass:** {ev.get('pass', False)}")
            
            ##### NEW
            # Display pass based on mode
            if getattr(self, "use_cot_generation", False):
                md_lines.append(f"- **COT Pass:** {cot_ev.get('pass', False)}")
            else:
                md_lines.append(f"- **Reference Pass:** {ev.get('pass', False)}")
                
            # --- Reference Evaluation Summary ---
            if ev and not getattr(self, "use_cot_generation", False):
                md_lines.append("- **Reference Evaluation:**")
                for k, v in ev.items():
                    if k not in {"pass", "suggested_revisions", "cot_evaluation"}:
                        md_lines.append(f"  - {k}: {v}")

            # --- COT Evaluation Summary (if exists) ---
            if cot_ev:
                md_lines.append("- **COT Evaluation:**")
                cot_score = cot_ev.get("reasoning_quality_score")
                if cot_score is not None:
                    md_lines.append(f"  - Reasoning quality score: {cot_score:.2f}")
                feedback = cot_ev.get("summary_feedback")
                if feedback:
                    md_lines.append(f"  - Feedback: {feedback.strip()}")
                cot_suggestions = cot_ev.get("suggested_revisions", [])
                if cot_suggestions:
                    md_lines.append("  - Suggested reasoning revisions:")
                    md_lines += [f"    - {s}" for s in cot_suggestions]
                    
            ####### END NEW
            else:
                # Revisions applied in the current round (if any were used)
                if i < len(all_revisions_history):
                    round_revs = all_revisions_history[i]
                    if round_revs:
                        md_lines.append("- **Revisions applied:**")
                        md_lines += [f"  - {r}" for r in round_revs]
                    else:
                        md_lines.append("- **Revisions applied:** _none_")
                else:
                    md_lines.append("- **Revisions applied:** _none_")

                # Suggested revisions from evaluation (for next round)
                sr = ev.get("suggested_revisions") or []
                if sr:
                    md_lines.append("- **Suggested revisions (for next round):**")
                    md_lines += [f"  - {s}" for s in sr if isinstance(s, str)]
                else:
                    md_lines.append("- **Suggested revisions (for next round):** _none_")
                md_lines.append("")

        # _write_text(os.path.join(self.out_dir, "summary_report.md"), "\n".join(md_lines))
        
        #### NEW
        report_path = os.path.join(self.out_dir, "summary_report.md")

        # Check if a report already exists
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                existing = f.read().strip()
        else:
            existing = ""

        final_report = "\n".join(md_lines)
        if existing:
            combined = existing + "\n\n---\n\n" + final_report
        else:
            combined = final_report

        _write_text(report_path, combined)

    def apply_rag(self, user_prompt: str) -> Tuple[str, Optional[str]]:
        print("Applying RAG to enhance the user prompt...")
        print(f"Original user prompt: {user_prompt}")
        
        print("Extracting attributes from the user prompt...")
        composer_name, music_form = self.rag.get_attributes(user_prompt)
        print(f"Extracted attributes: composer_name={composer_name}, music_form={music_form}")
        
        filtered_meta = self.rag.filter_music_data(composer_name, music_form)
        if len(filtered_meta) == 0:
            print("No matching music found in the database. Proceeding without RAG.")
            self.last_reference_abc = None
            return user_prompt, None
        
        print("Processing the reference music to obtain ABC notation...")
        reference_abc = self.rag.process_reference_music(filtered_meta, composer_name, music_form)
        updated_prompt = self.update_user_prompt(user_prompt, reference_abc)
        self.last_reference_abc = reference_abc
        return updated_prompt, reference_abc

    def update_user_prompt(self, user_prompt: str, reference_abc: str) -> str:
        updated_prompt = (
            f"{user_prompt} "
            f"Use the following as reference:{reference_abc}. "
            f"Follow the style and approximate phrase/length structure. "
            f"Ensure the output is in valid ABC notation. Remove anything that may break the format."
            # f"{user_prompt}\n\n"
            # f"Analyze the following reference music in ABC notation:\n"
            # f"{reference_abc}\n\n"
            # f"Then compose an original piece in ABC notation. "
            # f"The new composition must not reuse melodies, harmonies, rhythms, or motifs from the reference directly. Instead, emulate its expressive character, mood, and approximate phrase/length structure.\n"
            # f"Ensure the output is in valid ABC notation. Remove anything that may break the format."
        )
        return updated_prompt


def parse_args():
    ap = argparse.ArgumentParser(description="Chat->ABC automation with optional self-critique refinement")
    ap.add_argument("--out_dir", type=str, default="output",
                    help="Base directory to write ABC and XML files (default: output) — actual run is saved under a timestamped subfolder.")
    ap.add_argument("--prompt", type=str, default=None,
                    help="Override user prompt (otherwise uses CLI tail or default).")
    ap.add_argument("--reference_eval_agent", action="store_true",
                    help="Enable evaluation+refinement loop against reference.")
    ap.add_argument("--agent_max_rounds", type=int, default=3,
                    help="Max refinement rounds when eval agents are enabled (default: 3).")
    ap.add_argument("--use_cot_generation", action="store_true",
                    help="Ask LLM to return reasoning_steps along with the ABC (COT generation).")
    known, unknown = ap.parse_known_args()
    if known.prompt is None:
        tail = " ".join(unknown).strip()
        known.prompt = tail if tail else DEFAULT_USER_PROMPT
    return known


def main():
    args = parse_args()

    # Timestamped subdirectory inside args.out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_out_dir, exist_ok=True)

    vcomposer_client = VComposerClient(api_key=API_KEY, model=MODEL, out_dir=run_out_dir)
    ## added 
    vcomposer_client.use_cot_generation = getattr(args, "use_cot_generation", False)
    
    abc_text, obj, raw = vcomposer_client.ask_llm(
        user_prompt=args.prompt,
        reference_eval_agent=args.reference_eval_agent,
        agent_max_rounds=args.agent_max_rounds
    )

    # Save final ABC into the timestamped folder
    abc_out_path = os.path.join(run_out_dir, "final_result.abc")
    out_abs = vcomposer_client.save_abc(abc_text, abc_out_path)

    # Convert ABC -> MusicXML, writing to the same timestamped folder
    subprocess.run(
        ['python', 'utils/abc2xml.py', '-o', run_out_dir, abc_out_path],
        check=False
    )

    print("\n=== Response Summary ===")
    print(f"Run folder:\n  {run_out_dir}")
    print(f"ABC detected/saved to:\n  {out_abs}")
    print(f"XML converted/saved to:\n  {os.path.join(run_out_dir, 'final_result.xml')}")
    print(f"Summary report (JSON):\n  {os.path.join(run_out_dir, 'summary_report.json')}")
    print(f"Summary report (Markdown):\n  {os.path.join(run_out_dir, 'summary_report.md')}")
    if isinstance(obj, dict) and "detailed steps" in obj:
        print("\nDetailed Steps:")
        for b in obj["detailed steps"]:
            print(f"  {b}")


if __name__ == "__main__":
    main()
