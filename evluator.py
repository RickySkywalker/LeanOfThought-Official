import utils
from prover.lean.verifier import Lean4ServerScheduler
import warnings
import copy
from tqdm import tqdm

LEAN_PREFIX = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

class Evaluator:
    def __init__(self, 
                 unevalated_dataset, 
                 eval_ckpt_path,
                 max_concurrent_requests=128, 
                 timeout=30,
                 memory_limit=-1):
        
        self.lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=max_concurrent_requests, 
                                                    timeout=timeout, 
                                                    memory_limit=memory_limit, 
                                                    name='verifier')
        self.unevalated_dataset = unevalated_dataset
        self.max_concurrent_requests = max_concurrent_requests

        # ckpt path
        self.eval_ckpt_path = eval_ckpt_path
        self.valid_proof_ckpt_path = self.eval_ckpt_path + "/valid_proof"
        self.all_eval_result_ckpt_path = self.eval_ckpt_path + "/all_eval_result"
        utils.check_folder_exit(self.valid_proof_ckpt_path)
        utils.check_folder_exit(self.all_eval_result_ckpt_path)

    # This function will update eval_proof_cut but will not pop any theorem from the dataset unless some bug cases.
    # Under the parallel schedule, every finished theorems are expected to be poped rather than staying in the list
    # so we can always choose the top num_to_eval theorems to evaluate.
    # In addition, this function will also return the updated dataset_to_eval in the case of some bug.
    def _parallel_schedule_eval_result(self, 
                                       num_to_eval, 
                                       dataset_to_eval):
        ls_to_return = []
        for i in range(min(num_to_eval, len(dataset_to_eval))):
            try:
                curr = dataset_to_eval[i]
                curr_eval_proof_cut = curr["eval_proof_cut"]
                if curr_eval_proof_cut >= len(curr["Generated_proof"]):
                    warnings.warn("The proof cut is larger than the generated proof. May have unknown bug")
                    print(f"Current theorem idx:{curr['idx']}, current theorem cut: {curr_eval_proof_cut}")
                    continue
                curr_eval_data_record = {"idx": curr["idx"], 
                                         "Thm_idx": curr_eval_proof_cut,
                                         "Name": curr["Name"],
                                         "Statement": curr["Statement"],
                                         "Proof_to_eval": curr["Generated_proof"][curr_eval_proof_cut]}
                ls_to_return.append(curr_eval_data_record)
                dataset_to_eval[i]["eval_proof_cut"] += 1
            except Exception as e:
                print(f"Current theorem idx:{curr['idx']}, current theorem cut: {curr_eval_proof_cut}")
                print("Error in the parallel schedule eval result")
                print(e)
        return ls_to_return, dataset_to_eval

    # This function manages all the evaluation requests. It will submit the request to the scheduler and get output
    # It will saperate the output into: valid_proof and all_vaild_result
    # The thm_to_eval should have the same length as the self.max_concurrent_requests to max the usage of resources
    def submit_eval_request(self, thm_to_eval):
        eval_ls_to_submit = [LEAN_PREFIX + curr["Proof_to_eval"] for curr in thm_to_eval]
        request_id_list = self.lean4_scheduler.submit_all_request(eval_ls_to_submit)
        results = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        print("Finish get all request outputs")
        thm_evaluated = []
        valid_proof = []

        for i in range(len(eval_ls_to_submit)):
            curr_eval_result = results[i]
            curr_data_record = copy.deepcopy(thm_to_eval[i])

            # Add the correct eval result to valid_proof
            if curr_eval_result["pass"] == True:
                valid_proof_record = {"idx": curr_data_record["idx"],
                                      "Name": curr_data_record["Name"],
                                      "Statement": curr_data_record["Statement"],
                                      "Proof": curr_data_record["Proof_to_eval"]}
                valid_proof += [valid_proof_record]
            
            # Add the eval result to all_eval_result for record and change the name of dic
            curr_data_record["eval_result"] = curr_eval_result
            curr_proof = curr_data_record["Proof_to_eval"]
            del curr_data_record["Proof_to_eval"]
            curr_data_record["Generated_proof"] = curr_proof
            thm_evaluated += [curr_data_record]
        
        return valid_proof, thm_evaluated
            
    def write_ckpt(self, valid_proof, thm_evaluated):
        # Handle the valid proof, write it to the valid_proof_ckpt_path
        if len(valid_proof) > 0:
            for curr in valid_proof:
                curr_thm_name = curr["Name"]
                ckpt_path = self.valid_proof_ckpt_path + "/" + curr_thm_name + ".json"
                utils.write_to_json(ckpt_path, curr)
        
        # Handle the thm_evaluated, which is all eval result, save for reference, write it to the all_eval_result_ckpt_path
        for curr in thm_evaluated:
            # Make sure the path exis
            curr_thm_name = curr["Name"]
            ckpt_folder_path = self.all_eval_result_ckpt_path + "/" + curr_thm_name + ""
            utils.check_folder_exit(ckpt_folder_path)

            # write to file, it should be {thm_name}_{thm_idx}.json
            ckpt_file_path = f"{ckpt_folder_path}/{curr_thm_name}_{curr['Thm_idx']}.json"
            utils.write_to_json(ckpt_file_path, curr)

    def update_dataset(self, dataset_to_eval, valid_proof):

        valid_proof_idx_set = set([curr["idx"] for curr in valid_proof])
        for curr in dataset_to_eval:
            # If eval_proof_cut is equal to the total generated proof num, pop the theorem from the dataset
            if curr["eval_proof_cut"] >= len(curr["Generated_proof"]):
                dataset_to_eval.remove(curr)
                continue

            # If find a valid proof, then pop that theorem from the dataset
            if curr["idx"] in valid_proof_idx_set:
                dataset_to_eval.remove(curr)
                continue
        return dataset_to_eval

    def _remove_giveup_proof(self, dataset_to_eval):
        """
        This function will remove the giveup proof in the theorem list
        """

        processed_dataset = []
        for curr in dataset_to_eval:
            curr["Generated_proof"] = [curr_proof for curr_proof in curr["Generated_proof"] if "sorry" not in curr_proof and "admit" not in curr_proof]
            processed_dataset.append(curr)

        return processed_dataset

    # The eval_method means the way to evaluate the dataset. If saying "parallel", it will evaluate multiple different theorem at a time. 
    # If saying "sqeuential", it will evaluate one theorem for many proves at a time.
    def evaluate(self, 
                 begin_idx=0,
                 end_idx=-1, 
                 eval_method="parallel"):
        dataset_to_eval = self.unevalated_dataset[begin_idx:end_idx]

        # Remove the giveup proof
        dataset_to_eval = self._remove_giveup_proof(dataset_to_eval)

        # Init the progress bar
        total_thm_to_eval = len(dataset_to_eval)
        prog_bar = tqdm(total=len(dataset_to_eval))
        num_thm_to_eval = 0
        for curr in dataset_to_eval:
            num_thm_to_eval += len(curr["Generated_proof"])
        prog_bar_total = tqdm(total=num_thm_to_eval)
        vaild_proof_storage = []
        # initialize the eval proof cut, indicating how many proofs has been evluated in the Generated_proof ls.
        for i in range(len(dataset_to_eval)):
            dataset_to_eval[i]["eval_proof_cut"] = 0

        while len(dataset_to_eval) > 0:
            if eval_method == "parallel":
                ls_to_eval, dataset_to_eval = self._parallel_schedule_eval_result(self.max_concurrent_requests * 16, dataset_to_eval)
            else:
                raise NotImplementedError("The evaluation method is not implemented yet")

            # Evaluate the theorem
            print("submitting request")
            valid_proof, thm_evaluated = self.submit_eval_request(ls_to_eval)
            print("finish evaluate")
            vaild_proof_storage += valid_proof

            # write to ckpt
            self.write_ckpt(valid_proof, thm_evaluated)

            # Update the dataset
            dataset_to_eval = self.update_dataset(dataset_to_eval, valid_proof)

            # update the progress bar
            prog_bar.n = total_thm_to_eval - len(dataset_to_eval)
            prog_bar.refresh()
            prog_bar_total.update(len(thm_evaluated))
            if len(dataset_to_eval) == 0:
                break

        return vaild_proof_storage



            
            

            
