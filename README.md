# PRE_legal_NLP

## Legal further pre-training code: legal_pretraining.py

### Description of the code

**legal_pretraining.py is a pre-training routine made up to further pre-train generic models on legal datatsets. This code works with config file. Config.json is the template of a config file for this code. The config file allows to specify all the relevant parameters of the pre-training. This routine was built mostly based on the functions of the pre-training tutorial available on Hugging Face.**

### How to run the code

**To run the code, you have to fill all the variables of the config file and then you run the python code.**


## Courts decisions classification code : legal_finetuning.py

### Description of the code

**legal_finetuning.py is a fine-tuning routine made up to fine-tune models on legal classification tasks. To build the code, we customized the code used to run GLUE benchmark tasks (https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) by adding the estimations relevant micro and macro metrics and adapting the truncation of the tokenizer to the legal dataset.**

### How to run the code
**The code is built with class of arguments. Thus, to run the fine-tuning, you run a script with specified variables that you consider relevant.**
**Here is a template of a script to run a fine-tuning.**

`python legal_finetuning.py \
                                        --dataset_name  \
                                        --train_file  \
                                        --test_file  \
                                        --validation_file  \
                                        --model_name_or_path  \
                                        --output_dir  \
                                        --max_seq_length  \
                                        --per_device_train_batch_size  \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate  \
                                        --num_train_epochs  \
                                        --save_steps -1 \
                                        --fp16_opt_level O1 \
                                        |& tee output.log`
