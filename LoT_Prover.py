import Prover
import Corrector
from vllm import LLM
import utils
import torch
from prover.lean.verifier import Lean4ServerScheduler
from typing import List, Dict
import re   
from evluator import Evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel

DeepSeekProver_INST_SYS_PROMPT = """Follow these instructions carefully:
1. Provide a logically correct and rigorous Lean4 proof.
2. In the <Thought> section, include your detailed step-by-step reasoning.
3. In the <Output> section, provide only the final Lean4 proof or final result."""

DeepSeek_Corrector_SYS_PROMPT = """You are a helpful mathematical assistant specialized in formal theorem proving using Lean4. 
Your objectives:
1. Read and interpret the Lean4 theorem statement and any error messages.
2. **If a previous proof attempt was incorrect, analyze its exact mistakes and completely discard or rewrite the proof as needed.** 
3. **Avoid reusing incorrect proof structures or strategies unless explicitly validated as correct.**
4. **Address all error messages** by modifying the proof structure as needed.
5. Provide a detailed thought process in the <Thought> ... </Thought> section, but **only place the corrected Lean4 code in the <Output> ... </Output> section**.
6. **Ensure the new proof is logically valid and does not use `sorry`.**### Instruction:Below are some **correct Lean4 theorem proof examples** for your reference. Use them as guidance when constructing the revised proof. Ensure that your final proof aligns with these examples in terms of clarity, structure, and correctness."""


Lean4_HEADER = \
    """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

def extract_theorem_name(lean_code):
    match = re.search(r'theorem\s+([^\s(]+)', lean_code)
    if match:
        return match.group(1)
    return "test_theorem"

def print_valid_proof(thm_name, LongCoT_content, valid_proof):
    """
    This function is for printing the valid proof of the theorem.

    Args:
        thm_name: The name of the theorem.
        LongCoT_content: The Long CoT content for the proof.
        valid_proof: The valid proof of the theorem.
    """

    print(f"Find valid proof for `{thm_name}`")
    print(f"""The thinking process for the proof is:
<think>
{LongCoT_content}
</think>

The valid proof is:
```lean4
{valid_proof}
```""")

class LoT_Prover:

    def __init__(self, 
                 model_id: str,
                 scheduler: Lean4ServerScheduler,
                 terminators=None,
                 example_list=None, 
                 example_num=3):
        """
        This is the initialization function for LoT_Prover

        Args:
            model_id: The model id for the language model, should be a Long CoT model. LoT-Solver is suggested.
            scheduler: The scheduler for the Lean4 Server
            example_list: The list of few-shot examples to be proved, if `None` provided, will use the default examples.
            example_num: The number of few-shot examples used in the prove.
        """
        self.model_id = model_id
        self.model = LLM(model=model_id, dtype=torch.bfloat16, gpu_memory_utilization=0.95, enforce_eager=True)
        self.scheduler = scheduler
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if example_list == None:
            self.example_list = utils.read_from_json("./data/few_shot_examples.json")
        else:
            self.example_list = example_list

        if terminators == None:
            self.terminators = [self.tokenizer.eos_token_id]
        else:
            self.terminators = terminators
        self.example_num = example_num
        self.prover = Prover.Prove_writer(self.model, self.tokenizer, self.terminators, self.example_list, self.example_num)
        self.corrector = Corrector.Prove_writer(self.model, self.tokenizer, self.terminators)

    
    def run_prover_single_theorem(
            self, 
            Lean_statement: str, 
            NL_statement: str, 
            proof_num=4, 
            temperature=0.8, 
            top_p=0.8,
            repetition_penalty=1.0, 
            variable_tempreature=(-1.0),
            variable_top_p=(-1.0),
            max_tokens=2048,
            batch_size=4,
            print_result=False,
            system_prompt=DeepSeekProver_INST_SYS_PROMPT, 
            LongCoT_control=False,
            LongCoT_begin_sign="<Thought>",
            LongCoT_stop_sign="</Thought>", 
            output_begin_sign="<Output>",
            return_LongCoT_content=True,
        ):
        """
        This is the function that runs prover to generate whole proof for the single theorem. 

        Args:
            Lean_statement: The Lean4 statement for that prover need to prove, should start with `theorem` or `lemma`.
            NL_statement: The natural language statement of the Lean4 statement.
            proof_num: The number of proofs to be generated.
            temperature: The temperature for the generation process.
            top_p: The top_p for the generation process.
            repetition_penalty: The repetition penalty for the generation process.
            variable_tempreature: The lower-bound for temperature, if it is -1.0, will not use variable temperature.
            variable_top_p: The lower-bound for top-p, if it is -1.0, will not use variable top-p.
            max_tokens: The maximum number of tokens for the prompt + response.
            batch_size: The batch size for the generation process.
            print_result: Whether to print the result, default is `False`
            system_prompt: The system prompt for the generation process, default is `DeepSeekProver_INST_SYS_PROMPT`.
            LongCoT_control: Whether to use LongCoT control and expliitly add the Lean4 theorem statement at the end of Long CoT.
            LongCoT_begin_sign: The begin sign for the Long CoT content.
            LongCoT_stop_sign: The stop sign for the Long CoT content.
            output_begin_sign: The begin sign for the output content.
            return_LongCoT_content: Whether to return the Long CoT content.

        Returns:
            A list of complete lean4 theorem proofs in the format
            ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
             'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
        """

        # extract the theorem name
        theorem_name = extract_theorem_name(Lean_statement)

        # formulate the theorem record format that fits the prover
        curr_theorem_record = {
            "NL": NL_statement,
            "Name": theorem_name,
            "FL_statement": Lean_statement
        }

        # make generation using the prover
        if "lot-solver" in self.model_id.lower():
            prove_ls_to_return, Long_CoT_content = self.prover.generate_proof_singleThm(
                thm_record=curr_theorem_record,
                proof_num=proof_num,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                variable_tempreature=variable_tempreature,
                variable_top_p=variable_top_p,
                max_tokens=max_tokens,
                batch_size=batch_size,
                print_result=print_result,
                system_prompt=system_prompt, 
                llm_type="DeepSeek-Prover-SFT-OpenO1-Instruction_FineTune_1E-7_newTraining", 
                LongCoT_control=LongCoT_control,
                LongCoT_begin_sign=LongCoT_begin_sign,
                LongCoT_stop_sign=LongCoT_stop_sign,
                output_begin_sign=output_begin_sign,
                return_LongCoT_content=return_LongCoT_content
            )
        else:
            raise NotImplementedError

        return prove_ls_to_return, Long_CoT_content
    
    def run_lean_verification(
            self, 
            theorems_to_evaluate: List[str]
        ):
        """
        This is the function that runs the Lean4 verification for the given theorems in list. The result will returned in a list with the same order as the input list.

        Args:
            theorems_to_evaluate: The list of theorems to be evaluated. e.g. 
                ```json
                ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
                'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
                ```
        Returns:
            A list of evaluation results in the following format:
                ```json
                [{
                    "verified_code": str: the code sent to the verifier for verification
                    "errors": List[Dict]:
                    [{
                        "pos": {"line": xxx},
                        "data": str: the content of error
                    }, ...], 
                    "pass": bool, whether the verification passed
                }, ...]
                ```
        """

        request_id_list = self.scheduler.submit_all_request(theorems_to_evaluate)
        results = self.scheduler.get_all_request_outputs(request_id_list)
        return results
    
    def run_corrector_single_theorem(
            self, 
            Lean_statement: str, 
            NL_statement: str, 
            eval_results: List[Dict],
            correction_num=4,
            temperature=0.8,
            top_p=0.8,
            repetition_penalty=1.0,
            variable_tempreature=(-1.0),
            variable_top_p=(-1.0),
            max_tokens=8192,
            batch_size=4,
            print_result=False,
            system_prompt=DeepSeek_Corrector_SYS_PROMPT, 
            LongCoT_control=True, 
            LongCoT_begin_sign="<Thought>", 
            LongCoT_stop_sign="</Thought>",
            output_begin_sign="<Output>",
            return_LongCoT_content=True, 
    ):
        """
        This function is designed to run the corrector to correct wrong theorem for a single theorem with multiple wrong proofs demostrated in eval_results.

        Args:
            Lean_statement: The Lean4 statement for that prover need to prove, should start with `theorem` or `lemma`.
            NL_statement: The natural language statement of the Lean4 statement.
            eval_results: The list of evaluation results for the given theorems. The format should be:
                ```json
                [{
                    "verified_code": str: the code sent to the verifier for verification
                    "errors": List[Dict]:
                    [{
                        "pos": {"line": xxx},
                        "data": str: the content of error
                    }, ...], 
                    "pass": bool, whether the verification passed
                }, ...]
                ```
            correction_num: The number of corrections for each wrong result to be generated, should not be too large
            temperature: The temperature for the generation process.
            top_p: The top_p for the generation process.
            repetition_penalty: The repetition penalty for the generation process.
            variable_tempreature: The lower-bound for temperature, if it is -1.0, will not use variable temperature.
            variable_top_p: The lower-bound for top-p, if it is -1.0, will not use variable top-p.
            max_tokens: The maximum number of tokens for the prompt + response.
            batch_size: The batch size for the generation process.
            print_result: Whether to print the result, default is `False`
            system_prompt: The system prompt for the generation process, default is `DeepSeek_Corrector_SYS_PROMPT`.
            LongCoT_control: Whether to use LongCoT control and expliitly add the Lean4 theorem statement at the end of Long CoT.
            LongCoT_begin_sign: The begin sign for the Long CoT content.
            LongCoT_stop_sign: The stop sign for the Long CoT content.
            output_begin_sign: The begin sign for the output content.
            return_LongCoT_content: Whether to return the Long CoT content.

        Returns:
            A list of complete lean4 theorem proofs generated from correction, e.g.:
                ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
                'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
        """
        print(eval_results)
        theorem_records_to_test = []
        for i, curr in enumerate(eval_results):
            theorem_records_to_test.append({
                "idx": i, 
                "Thm_idx": i, 
                "NL": NL_statement,
                "Name": extract_theorem_name(Lean_statement),
                "Statement": Lean_statement,
                "Generated_proof": "NA",
                "eval_result": curr
            })


        # generate the correction using corrector
        if "lot-solver" in self.model_id.lower():
            corrected_theorems, Long_CoT_content = self.corrector.generation_correction_singleThm(
                thm_record_ls=theorem_records_to_test,
                correction_num=correction_num,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                variable_tempreature=variable_tempreature,
                variable_top_p=variable_top_p,
                max_tokens=max_tokens,
                batch_size=batch_size,
                print_result=print_result,
                system_prompt=system_prompt, 
                llm_type="DeepSeek-Prover-SFT-OpenO1-Corrector",
                LongCoT_control=LongCoT_control,
                LongCoT_begin_sign=LongCoT_begin_sign,
                LongCoT_stop_sign=LongCoT_stop_sign,
                output_begin_sign=output_begin_sign,
                return_LongCoT_content=return_LongCoT_content
            )
        else:
            raise NotImplementedError
        return corrected_theorems, Long_CoT_content
    
    def LoT_search_single_thm(
            self, 
            Lean_statement: str, 
            NL_statement: str, 
            whole_proof_gen_num=4, 
            correction_num_per_proof=2,
            max_search_depth=5,
            temperature=0.8,
            top_p=0.8,
            repetition_penalty=1.0,
            variable_tempreature=(-1.0),
            variable_top_p=(-1.0),
            max_tokens=8192,
            batch_size=4,
            correction_batch_size=2,
            print_result=False,
            LongCoT_control=False,
            LongCoT_begin_sign="<Thought>",
            LongCoT_stop_sign="</Thought>", 
            output_begin_sign="<Output>",
            return_LongCoT_content=True,
        ):
        """
        This function is for running MA-LoT search based on LoT-Prover and LoT-Corrector for a single theorem. The input should be a Lean 
        statement and the natural language statement of the theorem.

        Args:
            Lean_statement: The Lean4 statement for that prover need to prove, should start with `theorem` or `lemma`.
            NL_statement: The natural language statement of the Lean4 statement.
            whole_proof_gen_num: The number of proofs to be generated for the prover.
            correction_num_per_proof: The number of corrections to be generated for each wrong proof.
            max_search_depth: The maximum search depth for the search process.
            temperature: The temperature for the generation process.
            top_p: The top_p for the generation process.
            repetition_penalty: The repetition penalty for the generation process.
            variable_tempreature: The lower-bound for temperature, if it is -1.0, will not use variable temperature.
            variable_top_p: The lower-bound for top-p, if it is -1.0, will not use variable top-p.
            max_tokens: The maximum number of tokens for the prompt + response.
            batch_size: The batch size for the generation process.
            print_result: Whether to print the result, default is `False`
            LongCoT_control: Whether to use LongCoT control and expliitly add the Lean4 theorem statement at the end of Long CoT.
            LongCoT_begin_sign: The begin sign for the Long CoT content.
            LongCoT_stop_sign: The stop sign for the Long CoT content.
            output_begin_sign: The begin sign for the output content.
            return_LongCoT_content: Whether to return the Long CoT content.
        """

        thm_name = extract_theorem_name(Lean_statement)

        # use prover to generate the initial proofs with Long CoT guidance
        thm_prove_ls, LongCoT_content = self.run_prover_single_theorem(
            Lean_statement=Lean_statement,
            NL_statement=NL_statement,
            proof_num=whole_proof_gen_num,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            variable_tempreature=variable_tempreature,
            variable_top_p=variable_top_p,
            max_tokens=max_tokens,
            batch_size=batch_size,
            print_result=print_result,
            system_prompt=DeepSeekProver_INST_SYS_PROMPT, 
            LongCoT_control=LongCoT_control,
            LongCoT_begin_sign=LongCoT_begin_sign,
            LongCoT_stop_sign=LongCoT_stop_sign,
            output_begin_sign=output_begin_sign,
            return_LongCoT_content=return_LongCoT_content
        )

        # remove the give-up proof
        processed_thm_prove_ls = []
        processed_LongCoT_content = []

        for i, curr in enumerate(thm_prove_ls):
            if "sorry" not in curr and "admit" not in curr:
                processed_thm_prove_ls.append(curr)
                processed_LongCoT_content.append(LongCoT_content[i])

        eval_results = self.run_lean_verification(thm_prove_ls)

        for i, curr in enumerate(eval_results):
            if curr["pass"] == True:
                print_valid_proof(thm_name, processed_LongCoT_content[i], processed_thm_prove_ls[i])
                return {
                    "Name": thm_name, 
                    "Statement": Lean_statement, 
                    "Proof": processed_thm_prove_ls[i], 
                    "LongCoT": processed_LongCoT_content[i]
                }
        corrected_theorems = eval_results
            
        # if non of the proof is valid, then use the corrector to correct the proof
        for i in range(max_search_depth):
            corrected_theorems, corrected_LongCoT_content = self.run_corrector_single_theorem(
                Lean_statement=Lean_statement, 
                NL_statement=NL_statement, 
                eval_results=eval_results, 
                correction_num=correction_num_per_proof,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                variable_tempreature=variable_tempreature,
                variable_top_p=variable_top_p,
                max_tokens=max_tokens,
                batch_size=correction_batch_size,
                print_result=print_result,
                LongCoT_control=LongCoT_control,
                LongCoT_begin_sign=LongCoT_begin_sign,
                LongCoT_stop_sign=LongCoT_stop_sign,
                output_begin_sign=output_begin_sign,
                return_LongCoT_content=return_LongCoT_content
            )

            correction_eval_results = self.run_lean_verification(corrected_theorems)

            for i, curr in enumerate(correction_eval_results):
                if curr["pass"] == True:
                    print_valid_proof(thm_name, corrected_LongCoT_content[i], corrected_theorems[i])
                    return {
                        "Name": thm_name, 
                        "Statement": Lean_statement, 
                        "Proof": corrected_theorems[i], 
                        "LongCoT": corrected_LongCoT_content[i]
                    }

        print(f"Cannot find valid proof for `{thm_name}`")
        return None            



        



        

    

    