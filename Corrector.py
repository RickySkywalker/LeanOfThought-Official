from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random
from tqdm.auto import tqdm
import utils
from typing import List, Dict
from vllm.inputs import TokensPrompt
from vllm import LLM, SamplingParams
import os

DeepSeek_Corrector_SYS_PROMPT = """You are a helpful mathematical assistant specialized in formal theorem proving using Lean4. 
Your objectives:
1. Read and interpret the Lean4 theorem statement and any error messages.
2. **If a previous proof attempt was incorrect, analyze its exact mistakes and completely discard or rewrite the proof as needed.** 
3. **Avoid reusing incorrect proof structures or strategies unless explicitly validated as correct.**
4. **Address all error messages** by modifying the proof structure as needed.
5. Provide a detailed thought process in the <Thought> section, but **only place the corrected Lean4 code in the <Output> section**.
6. **Ensure the new proof is logically valid and does not use `sorry`.**### Instruction:Below are some **correct Lean4 theorem proof examples** for your reference. Use them as guidance when constructing the revised proof. Ensure that your final proof aligns with these examples in terms of clarity, structure, and correctness."""

Lean4_HEADER = \
    """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""


class Prove_writer:

    def __init__(self, 
                 model: LLM, 
                 tokenizer,
                 terminators):
        """
        The constructor for the Prover_writer class, the parameter list are as follows:
        Args:
            model: LLM model in the vllm package, we assume everything is set for the initialization
            terminators: the terminators for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminators = terminators


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

    # This function test whether there is a lean code block inside the text
    def _contains_lean_code_block(self, text):
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
    
    def _query_model_LongCoT_control(self, 
                                 prompt: str, 
                                 NL_statement: str, 
                                 FL_statement: str,
                                 max_tokens=1024,
                                 tempreature=0.9, 
                                 top_p=0.9,
                                 repetition_penalty=1.0,
                                 batch_size=4,
                                 LongCoT_stop_sign="</Thought>", 
                                 output_begin_sign="<Output>") -> List[str]:
        """
            This function is for query the model with given prompt and return the result string, we assume that
            the format is provided in the prompt. We will always return the format for prompt + response

            Args:
                prompt: the prompt for the model
                NL_statement: The Natural language statement of the theorem
                FL_statement: The formal language statement of the theorem
                max_tokens: The max token for the whole sequence (prompt + generated text)
                tempreature: the tempreature for the sampling
                top_p: the top_p for the sampling
                repetition_penalty: the repetition penalty for the sampling
                batch_size: the batch size for the sampling
                LongCoT_stop_sign: The stop sign for the CoT portion
                output_begin_sign: The sign to mark the beginning of the output
        """
        # Tokenize the original prompt
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        vllm_input_tokens: TokensPrompt = {"prompt_token_ids": tokenized_prompt}
        
        # Use a stop sign to restrict the CoT portion to 80% of the available tokens.
        stop_strs = [LongCoT_stop_sign]
        sampling_para = SamplingParams(
            temperature=tempreature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop_strs,
            max_tokens=int(max_tokens * 0.8),
            n=batch_size
        )
        outputs = self.model.generate(vllm_input_tokens, sampling_para)
        
        # For each output, add the header (with NL and FL) after the CoT stop sign and query again
        postCoT_vllm_input_tokens = []
        post_CoT_prompt_ls = []
        for curr in outputs[0].outputs:
            curr_input_str = f"""{prompt}{curr.text}
{LongCoT_stop_sign}

{output_begin_sign}
```lean4
{Lean4_HEADER}

/--{NL_statement}-/
{self._preprocess_theorem_statement(FL_statement)}"""
            curr_tokenized_prompt = self.tokenizer(curr_input_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            curr_tokens_prompt: TokensPrompt = {"prompt_token_ids": curr_tokenized_prompt}
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

    def _formulate_prompt_Corrector(self,
                                    NL: str,
                                    thm_name: str,
                                    FL_statement: str,
                                    eval_results: Dict,
                                    system_prompt=DeepSeek_Corrector_SYS_PROMPT,
                                    end_of_code="&"):
        """
        This function is designed for formulate prompt for LoT querying Corrector model. We only consider one
        round of correction and without example. We will add all BOS and EOS tokens if needed
        Args:
            NL: NL statement of the theorem
            thm_name: Name of the theorem
            FL_statement: Formal language statement of the theorem
            Wrong_FL_proof: the FL proof needed to be corrected, normally marked as "Generated_proof"
            eval_results: Same as "RL_data" in some cases, it contains the follwoing things
                "verified_code": str: the code sent to the verifier for verification
                "errors": List[Dict]: [{
                    "pos": {"line": xxx},
                    "data": str: the content of error
                }]
            end_of_code: The sign to mark the end of code

        Returns:
            prompt for querying the model with correction data
        """

        if FL_statement[-len("sorry"):] == "sorry":
            FL_statement = FL_statement[:-len("sorry")]

        error_msg = ""

        if "errors" in set(eval_results.keys()):
            for curr_error in eval_results["errors"]:
                error_msg += f"""```bash
line {curr_error["pos"]["line"]}

{curr_error["data"]}
```{end_of_code}

"""

        prompt = f"""<｜begin▁of▁sentence｜>{system_prompt}### Instruction: @ Natural language theorem statement:
{thm_name}
{NL}

@ Lean4 theorem statement:
```lean4
{FL_statement}
```{end_of_code}

@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
### Response:
<Thought>
Alright, I need to prove the theorem lean_workbook_plus_68493 using the Lean4 code. Here is my draft of the proof:

```lean4
{eval_results["verified_code"]}
```{end_of_code}

Let me test it in Lean4

Emmm, it seems the above proof is wrong.

Let me check the error messages.

OK, Here is the error messages:

{error_msg}
So, I will rethink a Lean4 proof following the steps

  1. Provide the natural language analysis for the theorem based on the Natural language theorem statement, Lean4 theorem statement, my previous proof and the error message.

  2. Draft the Lean4 tactics I should use to solve the problem

  3. Write the output Lean4 code.

Let me analysis the wrong Lean4 solution through the error messages"""
        return prompt

    def generation_correction_singleThm(
            self,
            thm_record_ls: List[Dict],
            correction_num=4,
            temperature=0.9,
            top_p=0.9,
            variable_top_p=(-1.0),
            variable_tempreature=(-1.0),
            max_tokens=8192,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            llm_type="DeepSeek-Prover-SFT-OpenO1-Corrector",
            system_prompt=DeepSeek_Corrector_SYS_PROMPT,
            end_of_code="&", 
            LongCoT_control=False,
            LongCoT_begin_sign="<Thought>",
            LongCoT_stop_sign="</Thought>", 
            output_begin_sign="<Output>",
            return_LongCoT_content=True,
        ) -> List:
        """
        This is the function to write corrections for single theorem, the input should a theorem's eval result list containing many
        wrong proofs of a theorem and some error messages

        Args:
            thm_record_ls: A list of Dict of eval results contains the following:
                [{
                    "idx": index of thetheorem,
                    "Thm_idx": the idx for verification,
                    "NL": NL statement of the
                    "Name": Name of the thm (every records should be the same,
                    "Statement": FL statement,
                    "Generated_proof": Proof generated by LLM
                    "eval_result": Dict of eval results, should contains the following:
                        {
                            "verified_code": str: the code sent to the verifier for verification
                            "errors": List[Dict]:
                            [{
                                "pos": {"line": xxx},
                                "data": str: the content of error
                            }]
                        }
                    "verified_code": Lean4 code directly put into the Lean evaluator for writing,
                    ...
                }, ...]

            correction_num: The correction number for every wrong proof, should not be too large
            batch_size: batch size for querying the model.
        """
        proof_ls = []
        Long_CoT_contents = []
        for curr_thm_record in thm_record_ls:
            for i in range(int(correction_num / batch_size)):
                if "deepseek" in llm_type.lower() and "prover" in llm_type.lower():
                    if "corrector" in llm_type.lower():
                        if "verified_code" not in curr_thm_record["eval_result"].keys():
                            UserWarning("The verified code is not in the eval result, please check the input")
                            continue
                        prompt = self._formulate_prompt_Corrector(curr_thm_record["NL"],
                                                                  curr_thm_record["Name"],
                                                                  curr_thm_record["Statement"],
                                                                  curr_thm_record["eval_result"],
                                                                  system_prompt=system_prompt,
                                                                  end_of_code=end_of_code)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                if print_result:
                    print("current prompt is")
                    print(prompt)

                # This codes are for variable hyper para
                if variable_tempreature != -1.0:
                    curr_temperature = random.uniform(variable_tempreature, temperature)
                else:
                    curr_temperature = temperature
                if variable_top_p != -1.0:
                    curr_top_p = random.uniform(variable_top_p, top_p)
                else:
                    curr_top_p = top_p

                try:
                    if not LongCoT_control:
                        curr_responses = self._query_model(
                            prompt,
                            max_tokens=max_tokens, 
                            tempreature=curr_temperature,
                            top_p=curr_top_p,
                            repetition_penalty=repetition_penalty,
                            batch_size=batch_size
                        )
                    else:
                        curr_responses = self._query_model_LongCoT_control(
                            prompt=prompt, 
                            NL_statement=curr_thm_record["NL"],
                            FL_statement=curr_thm_record["Statement"],
                            max_tokens=max_tokens,
                            tempreature=curr_temperature,
                            top_p=curr_top_p,
                            repetition_penalty=repetition_penalty,
                            batch_size=batch_size,
                            LongCoT_stop_sign=LongCoT_stop_sign,
                            output_begin_sign=output_begin_sign
                        )

                        if return_LongCoT_content:
                            for curr_response in curr_responses:
                                matches = re.findall(rf"{re.escape(LongCoT_begin_sign)}(.*?){re.escape(LongCoT_stop_sign)}", curr_response, re.DOTALL)
                                if matches:
                                    Long_CoT_contents += [matches[-1]]
                                else:
                                    Long_CoT_contents += [""]
                except Exception as e:
                    print(f"current error is {e}, skip current record")
                    continue

                if print_result:
                    print(f"{'#' * 20}\ncurrent temperature: {curr_temperature}")
                    for curr_response in curr_responses:
                        print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")

                for curr_response in curr_responses:
                    if self._contains_lean_code_block(curr_response):
                        curr_proof = self._extract_lean_code_blocks(curr_response)
                        proof_ls += [curr_proof]
        return proof_ls, Long_CoT_contents

    def preprocess_eval_datset(self,
                               valid_proof_path: str,
                               all_eval_result_path: str,
                               original_dataset_path: str) -> List[Dict]:
        """
        This function will preprocess all the valid_proofs, all_eval_results, and original dataset to formulate a datset that
        can be accepted by generate_correction_dataset. The evaluator by default is the one in Evaluator repo of Lean_of_Thought

        Args:
            valid_proof_path: The path for all the valid proofs
            all_eval_result_path: The all eval result generated by our Evaluator
            original_dataset_path: The original form of the dataset, the basis of our function
        """
        valid_proof_ls = utils.read_json_in_folder(valid_proof_path)
        all_eval_result_file_ls = os.listdir(all_eval_result_path)
        original_dataset = utils.read_from_json(original_dataset_path)
        original_datset_name_dict = {curr["Name"]: curr for curr in original_dataset}
        all_eval_result_ls = []
        for curr_file_path in all_eval_result_file_ls:
            curr_eval_result = utils.read_json_in_folder(f"{all_eval_result_path}/{curr_file_path}")
            all_eval_result_ls += [curr_eval_result]

        failed_theorems = []
        successed_thm_name_set = set([curr["Name"] for curr in valid_proof_ls])

        for curr_eval_result_ls in all_eval_result_ls:
            if len(curr_eval_result_ls) <= 0:
                continue
            else:
                curr_thm_name = curr_eval_result_ls[0]["Name"]
                if curr_thm_name not in successed_thm_name_set:
                    curr_thm_record = original_datset_name_dict[curr_thm_name]
                    curr_thm_record["eval_results_ls"] = curr_eval_result_ls
                    for i in range(len(curr_thm_record["eval_results_ls"])):
                        curr_thm_record["eval_results_ls"][i]["NL"] = curr_thm_record["Informal_statement"]
                    curr_thm_record["NL"] = curr_thm_record["Informal_statement"]
                    failed_theorems += [curr_thm_record]
        return failed_theorems

    def generate_correction_dataset(
            self,
            set_to_prove: List[Dict],
            correction_num=4,
            temperature=0.9,
            top_p=0.9,
            variable_top_p=(-1.0),
            variable_tempreature=(-1.0),
            max_tokens=8192,
            batch_size=4,
            repetition_penalty=1.0,
            ckpt_path="./Generated_proof_ckpt",
            llm_type="DeepSeek-Prover-SFT-OpenO1-Corrector",
            system_prompt=DeepSeek_Corrector_SYS_PROMPT,
            end_of_code="&", 
            LongCoT_control=False,
            LongCoT_begin_sign="<Thought>",
            LongCoT_stop_sign="</Thought>", 
            output_begin_sign="<Output>",
            return_LongCoT_content=True
        ) -> List[Dict]:
        """
        This function is designed for querying the correction data with given set of records of evaluation.

        Args:
            set_to_prove: List of dict that is used for eval:
                [{
                    "Name": Name of theorem,
                    "Statement": FL statement,
                    "Proof": (usually is sorry),
                    "NL": NL statement of the theorem, may also been tagged as Informal_statement
                    "eval_results_ls": List of dict for every detailed eval result
                        [{
                            "idx": index of thetheorem,
                            "Thm_idx": the idx for verification,
                            "NL": NL statement of the
                            "Name": Name of the thm (every records should be the same,
                            "Statement": FL statement,
                            "Generated_proof": Proof generated by LLM
                            "eval_result": Dict of eval results, should contains the following:
                                {
                                    "verified_code": str: the code sent to the verifier for verification
                                    "errors": List[Dict]:
                                    [{
                                        "pos": {"line": xxx},
                                        "data": str: the content of error
                                    }]
                                }
                            "verified_code": Lean4 code directly put into the Lean evaluator for writing,
                            ...
                        }]
                    ...
                }]
            correction_num: The correction number for every wrong proof, should not be too large
            batch_size: batch size for querying the model.
            temperature: The temperature for the sampling
            top_p: The top_p for the sampling
            variable_top_p: The lower bound for top_p in sampling, if it is -1.0, then it will not be used
            variable_tempreature: The lower bound for temperature in sampling, if it is -1.0, then it will not be used
            max_tokens: the max token for the whole sequence (prompt + generated text)
            repetition_penalty: the repetition penalty for the sampling
            ckpt_path: The path for the checkpoint
            llm_type: The type of the LLM model, this will determine the prompt for the model
            system_prompt: The system prompt for the model
            end_of_code: The sign to mark the end of code
            LongCoT_control: whether to use the Long CoT control method to add the theorem header after Long CoT
            LongCoT_begin_sign: the begin sign for the Long CoT
            LongCoT_stop_sign: the stop sign for the Long CoT
            output_begin_sign: the begin sign for the output


        """
        for i in tqdm(range(len(set_to_prove))):
            curr_return_result = {}
            curr_thm_record = set_to_prove[i]
            curr_prove_ls, Long_CoT_contents = self.generation_correction_singleThm(
                curr_thm_record["eval_results_ls"],
                correction_num=correction_num,
                temperature=temperature,
                top_p=top_p,
                variable_top_p=variable_top_p,
                variable_tempreature=variable_tempreature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                llm_type=llm_type,
                system_prompt=system_prompt, 
                end_of_code=end_of_code, 
                LongCoT_control=LongCoT_control,
                LongCoT_begin_sign=LongCoT_begin_sign,
                LongCoT_stop_sign=LongCoT_stop_sign, 
                output_begin_sign=output_begin_sign,
                return_LongCoT_content=return_LongCoT_content
            )

            for curr_key in list(curr_thm_record.keys()):
                if curr_key != "eval_results_ls":
                    curr_return_result[curr_key] = curr_thm_record[curr_key]
            curr_return_result["Generated_proof"] = curr_prove_ls
            if return_LongCoT_content:
                curr_thm_record["Long_CoT_content"] = Long_CoT_contents
            # curr_prove_ls += [curr_return_result]
            set_to_prove[i] = curr_return_result

            if ckpt_path != None:
                utils.write_to_json(f"{ckpt_path}/Generated_proof_{curr_thm_record['Name']}.json", curr_return_result)

        return set_to_prove