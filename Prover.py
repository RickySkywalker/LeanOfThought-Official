from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random
from tqdm.auto import tqdm
import utils
from typing import List, Dict
from vllm.inputs import TokensPrompt
from vllm import LLM, SamplingParams


MISTRAL_INSTRUCT_SYS_PROMPT = \
    """
You are a Lean4 expert who can write good Lean4 code based on Natural Language mathematical theorem statement and proof. Now you will be asked to write some Lean4 code based on Natural Language. 
[/INST]Sure, I would be happy to help you write Lean4 code based on Natural Language. Can you provide me with the Natural Language theorem statement and proof so that I can transfer them into Lean4 code.</s> [INST]
"""

Llama31_SYS_PROMPT = \
    """
Cutting Knowledge Date: June 2024
Today Date: 26 June 2024

You are a specialized mathematical assistant that converts natural language mathematical statements into formal Lean4 proofs."""

DeepSeekProver_HEADER_CoT = \
    """Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat"""

Qwen_SYS_PROMPT = "You are a mathematical assistant specialized in formal theorem proving using Lean4. Please integrate natural language reasoning with Lean4 to solve the problem below."

OpenO1_Qwen_SYS_PROMPT = "You are a mathematical assistant specialized in formal theorem proving using Lean4. Please complete the following Lean4 reasoning with explanatory comments preceding each line of code. You should write the answer using open o1 style Long CoT."

Lean4_HEADER = \
    """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

DeepSeek_R1_SYS_PROMPT = """Follow these instructions carefully:
1. Provide a logically correct and rigorous Lean4 proof.
2. In the <think> section, include your detailed step-by-step reasoning.
3. After the </think>, provide only the final Lean4 proof or final result."""

DeepSeekProver_INST_SYS_PROMPT = """Follow these instructions carefully:
1. Provide a logically correct and rigorous Lean4 proof.
2. In the <Thought> ... </Thought> section, include your detailed step-by-step reasoning.
3. In the <Output> ... </Output> section, provide only the final Lean4 proof or final result."""

OpenR1_SYS_PROMPT = """You are an expert in Lean4 theorem proving with exceptional strategic reasoning abilities. When solving problems, strictly follow this process:
1. First create a natural language proof draft explaining key insights and logical steps
2. Then analyze required Lean4 tactics, specifying exact syntax and their logical purpose

Present both sections clearly separated. Prioritize mathematical rigor while maintaining clarity."""

class Prove_writer:
    """
    This is the prover writer for wirting Lean4 code with vllm enhancement. For the version without vllm, please use Lean_writing.Prove_writer
    """

    def __init__(self, 
                 model: LLM, 
                 tokenizer,
                 terminators, 
                 example_ls, 
                 example_num=3):
        """
        The constructor for the Prover_writer class, the parameter list are as follows:
        Args:
            model: LLM model in the vllm package, we assume everything is set for the initialization
            terminators: the terminators for the model
            example_ls: the list of examples for the Lean4 query
            example_num: the number of examples to be used for the Lean4 query
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminators = terminators
        self.example_ls = example_ls
        self.example_num = example_num
    
    def _extract_lean_code_blocks(self, text: str):
        """
            This function extract the first lean code block from the text provided for normal LLMs, 
            if the query_type is "instruction" then the text should be only the response of the model, if the it is
            "code-completion" then the text should be the whole prompt + response of the model because the header of the 
            Lean4 code is provided in the prompt.

            Different from the original version, this function will always return the last code block in the text
        """
        lines = text.split('\n')
        inside_code_block = False
        code_blocks = []
        current_block = []

        for line in lines:
            if line.strip().startswith('```lean'):
                inside_code_block = True
                continue  # 开始标记行，跳过不收集
            elif "```" in line.strip() and inside_code_block:
                # 代码块结束，添加到代码块列表，并重置当前代码块
                code_blocks.append('\n'.join(current_block))
                current_block = []
                inside_code_block = False
                continue  # 结束标记行，跳过不收集

            if inside_code_block:
                current_block.append(line)

        try:
            if current_block != []:
                    code_blocks.append('\n'.join(current_block))
            return code_blocks[-1]
            # if (("deepseek" in llm_type.lower() and 'prover' in llm_type.lower()) or
            #         ("theoremllama" in llm_type.lower()) or
            #         ("theoremqwen" in llm_type.lower())
            # ):
            #     if current_block != []:
            #         code_blocks.append('\n'.join(current_block))
            #     return code_blocks[-1]
            # else:
            #     return code_blocks[0]
        except Exception:
            print(f"current result has unexcepted problem, generated text is:{text}")
            return -1

    def _contains_lean_code_block(self, text):
        """
        This function test whether there is a lean code block inside the text
        """
        # 使用正则表达式匹配以 ```lean 开始，以 ``` 结束的代码块
        # re.DOTALL 使得 . 匹配包括换行符在内的所有字符
        pattern = r'```lean.*?```'
        match = re.search(pattern, text, re.DOTALL)
        return match is not None
    
    def _preprocess_theorem_statement(self,
                                      input_statement: str):
        while not input_statement.endswith(":="):
            input_statement = input_statement[:-1]

        if input_statement.endswith(":="):
            input_statement = input_statement[:-len(":=")]
        input_statement += ":= by"
        return input_statement
    
    def _query_model_LongCoT_control(
            self, 
            prompt: str, 
            NL_statement: str, 
            FL_statement: str,
            max_tokens=1024,
            tempreature=0.9, 
            top_p=0.9,
            repetition_penalty=1.0,
            batch_size=4, 
            LongCoT_stop_sign="</Thought>", 
            output_begin_sign="<Output>"
    ) -> List[str]:
        """
        This function is for query the model with Long CoT control and return the result string, we assume that the format 
        is provided in the prompt. We will always return the format for prompt + response

        Args:
            prompt: the prompt for the model
            max_tokens: The max token for the whole sequence (prompt + generated text)
            tempreature: the tempreature for the sampling
            top_p: the top_p for the sampling
            repetition_penalty: the repetition penalty for the sampling
            batch_size: the batch size for the sampling
            LongCoT_stop_sign: the stop sign for the Long CoT, used to control the end of Long CoT

        Returns:
            A list of generated text for the model
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        vllm_input_tokens: TokensPrompt = {
            "prompt_token_ids": tokenized_prompt
        }
        stop_strs = [LongCoT_stop_sign]
        sampling_para = SamplingParams(
            temperature=tempreature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_strs,
            max_tokens=int(max_tokens * 0.8),   # We will use 80% of the max_tokens for the Long CoT
            n=batch_size
        )
        outputs = self.model.generate(vllm_input_tokens, sampling_para)

        postCoT_vllm_input_tokens = []
        post_CoT_prompt_ls = []

        for curr in outputs[0].outputs:
            curr_input_str = f"""{prompt}{curr.text}
{LongCoT_stop_sign}

{output_begin_sign}

Completed Lean 4 code with explanatory comments preceding each line of code:
```lean4
{Lean4_HEADER}

/--{NL_statement}-/
{self._preprocess_theorem_statement(FL_statement)}"""
            curr_tokenized_prompt = self.tokenizer(curr_input_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            curr_tokens_prompt : TokensPrompt = {
                "prompt_token_ids": curr_tokenized_prompt
            }
            postCoT_vllm_input_tokens.append(curr_tokens_prompt)
            post_CoT_prompt_ls.append(curr_input_str)
        
        postCoT_sampling_para = SamplingParams(
            temperature=tempreature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            n=1
        )

        postCoT_outputs = self.model.generate(postCoT_vllm_input_tokens, postCoT_sampling_para)

        output_strings = []
        
        for i in range(len(post_CoT_prompt_ls)):
            curr_output_string = post_CoT_prompt_ls[i] + postCoT_outputs[i].outputs[0].text
            output_strings.append(curr_output_string)
        return output_strings



    def _query_model(self, 
                     prompt: str, 
                     max_tokens=1024,
                     tempreature=0.9, 
                     top_p=0.9,
                     repetition_penalty=1.0,
                     batch_size=4):
        """
            This function is for query the model with given prompt and return the result string, we assume that 
            the format is provided in the prompt. We will always return the format for prompt + response

            Args:
                prompt: the prompt for the model
                max_tokens: The max token for the whole sequence (prompt + generated text)
                tempreature: the tempreature for the sampling
                top_p: the top_p for the sampling
                repetition_penalty: the repetition penalty for the sampling
                batch_size: the batch size for the sampling
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        vllm_input_tokens: TokensPrompt = {
            "prompt_token_ids": tokenized_prompt
        }
        sampling_para = SamplingParams(
            temperature=tempreature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            n=batch_size
        )
        outputs = self.model.generate(vllm_input_tokens, sampling_para)
        output_strings = [prompt + curr.text for curr in outputs[0].outputs]
        return output_strings
    
    def _formulate_prompt_DeepSeekProver(self,
                                         NL,
                                         FL_statement):
        """
        This is the function for formulate the prompt for DeepSeek-Prover when doing code-completion

        Args:
            NL: the natural language statement
            FL_statement: the formal language statement
        """
        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        selected_examples = random.sample(self.example_ls, self.example_num)
        prompt = ""

        # Write examples
        for curr_example in selected_examples:
            prompt += (DeepSeekProver_HEADER_CoT + "\n"
                                                   f"/--{curr_example['NL']}-/\n"
                                                   f"{curr_example['FL']}\n"
                                                   "```&\n\n")

        prompt += (DeepSeekProver_HEADER_CoT + "\n"
                                               f"/--{NL}-/\n"
                                               f"{FL_statement}\n")

        while (prompt.endswith(" ") or prompt.endswith("\n")):
            if prompt.endswith(" "):
                prompt = prompt[:-1]
            else:
                prompt = prompt[:-len("\n")]

        if (not prompt.endswith("by")):
            prompt = prompt + " by"

        if not prompt.endswith("\n"):
            prompt = prompt + "\n"

        if not prompt.startswith("<｜begin▁of▁sentence｜>"):
            prompt = "<｜begin▁of▁sentence｜>" + prompt
        return prompt
    
    def _formulate_instruction_prompt_DeepSeekProver(self,
                                                     NL,
                                                     thm_name,
                                                     FL_statement,
                                                     system_prompt=DeepSeekProver_INST_SYS_PROMPT):
        """
        This function formulates the instruction prompt for DeepSeek-Prover
        """
        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        selected_examples = random.sample(self.example_ls, self.example_num)
        prompt = f"""<｜begin▁of▁sentence｜>{system_prompt}### Instruction:You will receive several Lean4 problems. For each:
- **Use** a step-by-step solution internally in <Thought> ... </Thought>.
- **Do not** reveal your chain of thought outside the <Thought> ... </Thought> block.
- **Ensure** the final Lean4 code or final result is placed **only** in <Output> ... </Output>.
"""

        for curr_example in selected_examples:
            prompt += f"""@ Natural language theorem statement:
{curr_example["Name"]}
{curr_example['Informal_statement']}

@ Lean4 theorem statement:
```lean4
{self._preprocess_theorem_statement(curr_example["Statement"])}
```

@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
```lean4
{Lean4_HEADER}

/--{curr_example["Informal_statement"]}-/
{curr_example["Commented_proof"]}
```&
{"=" * 20}
"""

        prompt += f"""@ Natural language theorem statement:
{thm_name}:
{NL}

@ Lean4 theorem statement:
```lean4
{self._preprocess_theorem_statement(FL_statement)}
```&

@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
### Response:
<Thought>
Alright, I should do the following:

  1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.

  2. Draft the Lean4 tactics I should use to solve the problem

  3. Write the output Lean4 code.

The user also asks that I should avoid using the keyword `sorry` to give up the proof, so I will not write it in my Lean4 code.

The `{thm_name}` can be proofed by"""
        # # make sure we only end the line for one time
        # while prompt.endswith("\n\n"):
        #     prompt = prompt[:-len("\n")]

        return prompt

    def _formulate_prompt_DeepSeekProver_OpenR1(
        self, 
        NL, 
        thm_name, 
        FL_statement, 
        system_prompt=OpenR1_SYS_PROMPT
    ):
        """
        This function formulate the prompt for DeepSeek-R1-Distilled
        """
        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        selected_examples = random.sample(self.example_ls, self.example_num)

        prompt = f"""<｜begin▁of▁sentence｜>{system_prompt}
### Instruction: Please solve this Lean4 problem by completing both sections below:

NL Proof Draft (Natural Language)
Explain the proof strategy using mathematical reasoning and high-level steps. Consider:
- Key lemmas/theorems to apply
- Structural decomposition approaches
- Critical logical dependencies
- Potential proof patterns/methods

Lean4 Tactics Analysis (Technical Specification)
Identify concrete tactics needed to implement the proof, including:
1. Tactic name and syntax template
2. Purpose within proof context 
3. Expected goal state before/after application
4. Alternative tactics considered"""

        for curr_example in selected_examples:
            prompt += f"""@ Natural language theorem statement:
{curr_example["Name"]}
{curr_example['Informal_statement']}

@ Lean4 theorem statement:
```lean4
{curr_example["Statement"]} by
```

@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
```lean4
{Lean4_HEADER}

/--{curr_example["Informal_statement"]}-/
{curr_example["Commented_proof"]}
```&
{"=" * 20}
"""
        prompt += f"""@ Natural language theorem statement:
{thm_name}:
{NL}

@ Lean4 theorem statement:
```lean4
{self._preprocess_theorem_statement(FL_statement)}
```&


@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
### Response:
<think>"""

        return prompt
    
    def _formulate_prompt_DeepSeekR1_distilled_Qwen(self,
                                                    NL,
                                                    thm_name,
                                                    FL_statement,
                                                    system_prompt=DeepSeek_R1_SYS_PROMPT):
        """
        This function formulate the prompt for DeepSeek-R1-Distilled
        """
        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        selected_examples = random.sample(self.example_ls, self.example_num)

        prompt = f"""<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>You will receive several Lean4 problems. For each:
- **Use** a step-by-step solution internally in <think>.
- **Do not** reveal your chain of thought outside the <think> block.
- **Ensure** the final Lean4 code or final result is placed **only** after your thinking.

"""

        for curr_example in selected_examples:
            prompt += f"""@ Natural language theorem statement:
{curr_example["Name"]}
{curr_example['Informal_statement']}

@ Lean4 theorem statement:
```lean4
{curr_example["Statement"]}
```

@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
```lean4
{Lean4_HEADER}

/--{curr_example["Informal_statement"]}-/
{curr_example["Commented_proof"]}
```&
{"=" * 20}
"""
        prompt += f"""@ Natural language theorem statement:
{thm_name}:
{NL}

@ Lean4 theorem statement:
```lean4
{self._preprocess_theorem_statement(FL_statement)}
```&


@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:<｜Assistant｜>
<think>Okay, I should do the following:

  1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.

  2. Draft the Lean4 tactics I should use to solve the problem

  3. Write the output Lean4 code.

The user also asks that I should avoid using the keyword `sorry` to give up the proof, so I will not write it in my Lean4 code.

The `{thm_name}` can be proofed by"""

        return prompt

    def generate_proof_singleThm(self, 
                                 thm_record,
                                 proof_num=128, 
                                 temperature=0.9,
                                 top_p=0.9,
                                 variable_top_p=(-1.0),
                                 variable_tempreature=(-1.0),
                                 max_tokens=2048,
                                 batch_size=4,
                                 repetition_penalty=1.0,
                                 print_result=True,
                                 LongCoT_control=False,
                                 LongCoT_begin_sign="<Thought>",
                                 LongCoT_stop_sign="</Thought>", 
                                 output_begin_sign="<Output>",
                                 return_LongCoT_content=True,
                                 llm_type="DeepSeek-Prover",
                                 system_prompt="You are a Lean4 expert who can write good Lean4 code based on natural language mathematical theorem and proof"):
        """
        This function ask the LLM in Prover_writer class to write the proof based on the thm_record, 
        this is most of the model-specific setting take place.

        Args:
            thm_record: the record for the theorem of the following format:
                {
                    "NL": <Natural Language version of theorem statement and prove>,
                    "Name": <Name of current theorem>,
                    "FL_statement": <Formal Language version of theorem statement>, ...
                }
            proof_num: the number of proofs to be generated
            temperature: the temperature for the sampling
            top_p: the top_p for the sampling
            variable_top_p: The lower bound for top_p in sampling, if it is -1.0, then it will not be used
            variable_tempreature: The lower bound for temperature in sampling, if it is -1.0, then it will not be used
            max_new_tokens: the max token for the whole sequence (prompt + generated text)
            batch_size: the batch size for the sampling
            repetition_penalty: the repetition penalty for the sampling
            print_result: whether to print the result
            system_prompt: the system prompt for the model
            LongCoT_control: whether to use the Long CoT control method to add the theorem header after Long CoT
            LongCoT_stop_sign: the stop sign for the Long CoT
            output_begin_sign: the begin sign for the output

        Returns:

            A list of complete lean4 theorem proofs in the format
            ['theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
             'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...]
        """

        proof_ls = []
        Long_CoT_contents = []

        for i in range(int(proof_num / batch_size)):

            if "deepseek" in llm_type.lower() and "prover" in llm_type.lower():
                if "direct-code-filling" in llm_type.lower():
                    prompt = self._formulate_prompt_DeepSeekProver(
                        thm_record["NL"], 
                        thm_record["FL_statement"]
                    )
                elif "instruction" in llm_type.lower() and "openr1" not in llm_type.lower():
                    prompt = self._formulate_instruction_prompt_DeepSeekProver(
                        thm_record["NL"], 
                        thm_record["Name"],
                        thm_record["FL_statement"],
                        system_prompt=system_prompt
                    )
                elif "instruction" in llm_type.lower() and "openr1" in llm_type.lower():
                    prompt = self._formulate_prompt_DeepSeekProver_OpenR1(
                        thm_record["NL"], 
                        thm_record["Name"],
                        thm_record["FL_statement"],
                        system_prompt=system_prompt
                    )
                else:
                    prompt = self._formulate_prompt_DeepSeekProver(
                        thm_record["NL"], 
                        thm_record["FL_statement"]
                    )
            elif "deepseek" in llm_type.lower() and "r1" in llm_type.lower() and "qwen" in llm_type.lower():
                prompt = self._formulate_prompt_DeepSeekR1_distilled_Qwen(
                    thm_record["NL"],
                    thm_record["Name"],
                    thm_record["FL_statement"],
                    system_prompt=system_prompt
                )
            else:
                raise NotImplementedError


            if print_result:
                print("current prompt is")
                print(prompt)

            if variable_tempreature != -1.0:
                curr_temperature = random.uniform(variable_tempreature, temperature)
            else:
                curr_temperature = temperature

            if variable_top_p != -1.0:
                curr_top_p = random.uniform(variable_top_p, top_p)
            else:
                curr_top_p = top_p

            if not LongCoT_control:
                # Query the model without the Long CoT control
                curr_responses = self._query_model(
                    prompt, 
                    max_tokens=max_tokens, 
                    tempreature=curr_temperature,
                    top_p=curr_top_p,
                    repetition_penalty=repetition_penalty,
                    batch_size=batch_size
                )
            else:
                # Query the model with the Long CoT control
                curr_responses = self._query_model_LongCoT_control(
                    prompt, 
                    thm_record["NL"], 
                    thm_record["FL_statement"],
                    max_tokens=max_tokens,
                    tempreature=curr_temperature,
                    top_p=curr_top_p,
                    repetition_penalty=repetition_penalty,
                    batch_size=batch_size,
                    LongCoT_stop_sign=LongCoT_stop_sign,
                    output_begin_sign=output_begin_sign
                )

                # extract long CoT result if indicated
                if return_LongCoT_content:
                    for curr_response in curr_responses:
                        matches = re.findall(rf"{re.escape(LongCoT_begin_sign)}(.*?){re.escape(LongCoT_stop_sign)}", curr_response, re.DOTALL)
                        if matches:
                            Long_CoT_contents += [matches[-1]]
                        else:
                            Long_CoT_contents += [""]


            if print_result:
                print(f"{'#' * 20}\ncurrent temperature: {curr_temperature}")
                for curr_response in curr_responses:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")

            for curr_response in curr_responses:
                if self._contains_lean_code_block(curr_response):
                    curr_proof = self._extract_lean_code_blocks(curr_response)
                    proof_ls += [curr_proof]
        return proof_ls, Long_CoT_contents

    def generate_proof_dataset(self, 
                               set_to_prove: List[Dict], 
                               proof_num=128, 
                               max_tokens=2048,
                               temperature=0.9,
                               top_p=0.9,
                               variable_top_p=(-1.0),
                               variable_tempreature=(-1.0),
                               batch_size=4,
                               repetition_penalty=1.0,
                               LongCoT_control=False,
                               LongCoT_begin_sign="<Thought>",
                               LongCoT_stop_sign="</Thought>", 
                               output_begin_sign="<Output>",
                               return_LongCoT_content=True,
                               print_result=True,
                               ckpt_path="./Generated_proof_ckpt",
                               llm_type="Llama3-Instruct",
                               system_prompt="You are a Lean4 expert who can write good Lean4 code based on natural language mathematical theorem and proof"
        ) -> List[Dict]:
        """
        This is the function for generating the proof for the whole dataset and return the updated dataset with generated proof

        Args:
            set_to_prove: The list of theorem record in dict format to be proved. The theorem record should be the following format
                {
                    "NL": <Natural Language version of theorem statement and prove>,
                    "Name": <Name of current theorem>,
                    "FL_statement": <Formal Language version of theorem statement>, ...
                }
            proof_num: the number of proofs to be generated for each theorem record
            temperature: the temperature for the sampling
            top_p: the top_p for the sampling
            variable_top_p: The lower bound for top_p in sampling, if it is -1.0, then it will not be used
            variable_tempreature: The lower bound for temperature in sampling, if it is -1.0, then it will not be used
            max_tokens: the max token for the whole sequence (prompt + generated text)
            batch_size: the batch size for the sampling
            repetition_penalty: the repetition penalty for the sampling
            ckpt_path: the path for the checkpoint
            llm_type: the type of the LLM model. This will determine the prompt for the model
            system_prompt: the system prompt for the model
            LongCoT_control: whether to use the Long CoT control method to add the theorem header after Long CoT
            LongCoT_begin_sign: the begin sign for the Long CoT
            LongCoT_stop_sign: the stop sign for the Long CoT
            output_begin_sign: the begin sign for the output

        Returns:
            The updated list of dict dataset with the generated proof, each data record should have the following format:
                {
                    "NL": <Natural Language version of theorem statement and prove>,
                    "Name": <Name of current theorem>,
                    "FL_statement": <Formal Language version of theorem statement>,
                    "Generated_proof":
                        [
                            'theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  -- aesop?\n  ring',
                            theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n  rfl', ...
                        ]
                }
        """

        for i in tqdm(range(len(set_to_prove))):
            curr_thm_record = set_to_prove[i]
            curr_prove_list, LongCoT_contents = self.generate_proof_singleThm(
                curr_thm_record,
                proof_num=proof_num,
                temperature=temperature,
                top_p=top_p,
                variable_top_p=variable_top_p,
                variable_tempreature=variable_tempreature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                llm_type=llm_type,
                system_prompt=system_prompt, 
                print_result=print_result,
                LongCoT_control=LongCoT_control,
                LongCoT_stop_sign=LongCoT_stop_sign,
                output_begin_sign=output_begin_sign, 
                return_LongCoT_content=return_LongCoT_content,
                LongCoT_begin_sign=LongCoT_begin_sign
            )

            curr_thm_record["Generated_proof"] = curr_prove_list
            if return_LongCoT_content:
                curr_thm_record["Long_CoT_content"] = LongCoT_contents
            set_to_prove[i] = curr_thm_record

            if ckpt_path != None:
                utils.write_to_json(f"{ckpt_path}/Generated_proof_{curr_thm_record['Name']}.json", curr_thm_record)

        return set_to_prove