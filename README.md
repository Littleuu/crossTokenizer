# SeDi

This is the official code repository for the paper “Bridging the Tokenizer Gap: Semantics and Distribution-aware Knowledge Transfer for Unbiased Cross-Tokenizer Distillation”, accepted to AAAI 2026.
We propose **SeDi**, a semantics- and distribution-aware knowledge transfer framework for cross-tokenizer distillation.

# Experimental Environment Setup

You can install the required dependencies using either of the following methods:

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yml
```

# Datasets

We evaluate our method on three tasks: instruction following, code generation, and math reasoning. 


|  Method   |  Train  | Test  |
| :----: | :----: | :----: | 
| **Instruction Following**  | Dolly | Snist, Unist, Self-Inst, Vicuna | 
| **Code Generation** | CodeM | HumanEval |
| **Math Reasoning** | MetaMath | Orca, GSM8K, Math |


# Fine-tuning Teacher and Student Models

If a fine-tuned teacher model is already available, you can directly use it by updating the relevant information in the script file. If no fine-tuned teacher model exists, we recommend fine-tuning one on your target dataset using the following: ```bash finetune_teacher.sh```. 

- Change ```MODEL_TYPE``` to your teacher model type.

- Change ```CKPT_PATH``` to the path of your pre-trained model. 

- If you use LoRA fine-tuning, add the following parameters:

```
--peft lora --peft-lora-r 256 --peft-lora-alpha 8 --peft-lora-dropout 0.1 \
```

Similarly, we recommend fine-tuning the student model for 3 epochs before distillation using: ```bash finetune_student.sh```. 

- Also update ```MODEL_TYPE``` and ```CKPT_PATH``` accordingly.



# Running Distillation

We support five baseline methods as well as our proposed SeDi method. You can run each method with the following scripts:

|  Method   |  Scripts  | 
| :----: | :----: | 
| **MinED**  | bash minedit.sh | 
| **CDM** | bash cdm.sh | 
| **ULD** | bash uld.sh | 
| **MultiLevelOT** | bash multi_level_OT.sh | 
| **DSKD** | bash dskd.sh | 
| **SeDi** | bash sedi.sh | 


