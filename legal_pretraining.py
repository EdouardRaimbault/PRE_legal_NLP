import collections
import datetime
import json
import logging
import os
import time

from typing import Dict
from plot_lr import plot_metric_with_steps,clear_plot
import numpy as np
from torch.optim import AdamW
from knockknock import slack_sender

from datasets import Dataset
from transformers import (
   AutoModelForMaskedLM,
   CamembertTokenizer,
   CamembertForMaskedLM,
   RobertaTokenizer,
   RobertaForMaskedLM,
   CamembertForCausalLM,
   RobertaForCausalLM,
   FlaubertTokenizer,
   FlaubertWithLMHeadModel,
   AutoModelForCausalLM,
   DataCollatorForLanguageModeling,
   TrainingArguments,
   Trainer,
   TrainerCallback,
   TrainerControl,
   TrainerState,
   get_polynomial_decay_schedule_with_warmup
   
   )



class LearningRatePlotCallback(TrainerCallback):


   def on_step_end(self, args, state, control, **kwargs):
       output_file_path = os.path.join(args.output_dir, "learning_rate.jsonl")
       current_step = state.global_step
       optimizer_learning_rate = kwargs['optimizer'].param_groups[0]['lr']
       with open(output_file_path, 'a+') as output_file:
           output_file.write(json.dumps(
               {
                   "step": current_step,
                   "learning_rate": optimizer_learning_rate
               }
           ) + "\n")

class LossPlotCallback(TrainerCallback):
   """
   Callback called at the end of training to plot the loss function
   """
   def on_train_end(
           self,
           args: TrainingArguments,
           state: TrainerState,
           control: TrainerControl,
           **kwargs
   ):
       train_losses = [step.get('loss') for step in state.log_history if "loss" in step.keys()]
       train_steps = [step.get('step') for step in state.log_history if "loss" in step.keys()]


       eval_losses = [step.get('eval_loss') for step in state.log_history if "eval_loss" in step.keys()]
       eval_steps = [step.get('step') for step in state.log_history if "eval_loss" in step.keys()]


       plot_metric_with_steps(
           train_losses,
           train_steps,
           os.path.join(args.output_dir, "train_loss.png"))

       clear_plot()

       plot_metric_with_steps(
           eval_losses,
           eval_steps,
           os.path.join(args.output_dir, "eval_loss.png"))





def generate_jsonl_lines_from_single_file(file_path: str):
   with open(file_path, 'r+') as input_file:
       for line in input_file:
           yield json.loads(line)




def generate_jsonl_lines(root_path: str, recursive: bool = False):
   if not os.path.exists(root_path):
       raise FileNotFoundError(f" : File or directory not found at {root_path}")
   if os.path.isfile(root_path):
       yield from generate_jsonl_lines_from_single_file(root_path)
   elif not recursive:
       for file_name in os.listdir(root_path):
           if file_name.endswith(".jsonl"):
               yield from generate_jsonl_lines_from_single_file(os.path.join(root_path, file_name))
   else:
       for jsonl_file in generate_files_in_dir(root_path, ".jsonl"):
           yield from generate_jsonl_lines_from_single_file(jsonl_file)


def generate_files_in_dir(root_dir: str, extension: str = None):
   if not os.path.exists(root_dir):
       raise FileNotFoundError(f" : Directory not found at {root_dir}")
   if extension:
       for dp, dn, filenames in os.walk(root_dir):
           for input_file in filenames:
               if input_file.endswith(extension):
                   yield os.path.join(dp, input_file)
   else:
       for dp, dn, filenames in os.walk(root_dir):
           for input_file in filenames:
               yield os.path.join(dp, input_file)


def read_json_file(file_path: str) -> Dict:
   """
   Read a json document from a given file path


   Args:
       file_path: path to json file


   Returns:
       Dict of json encoded document
   """
   with open(file_path, "r+") as input_file:
       parsed = json.loads(input_file.read())
   return parsed




def tokenize_function(examples, tokenizer):
   result = tokenizer(examples['text'])
   if tokenizer.is_fast:
       result['word_ids'] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
   return result




def group_texts(examples, chunk_size=768):
   # Concatenate all texts
   concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
   # Compute length of concatenated texts
   total_length = len(concatenated_examples[list(examples.keys())[0]])
   # We drop the last chunk if it's smaller than chunk_size
   total_length = (total_length // chunk_size) * chunk_size
   # Split by chunks of max_len
   result = {
       k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
       for k, t in concatenated_examples.items()
   }
   # Create a new labels column
   result["labels"] = result["input_ids"].copy()
   return result




def whole_word_masking_data_collator(features, tokenizer, data_collator):


   # TODO : see if this is the same everywhere
   wwm_probability = 0.2


   for feature in features:
       word_ids = feature.pop("word_ids")


       # Create a map between words and corresponding token indices
       mapping = collections.defaultdict(list)
       current_word_index = -1
       current_word = None
       for idx, word_id in enumerate(word_ids):
           if word_id is not None:
               if word_id != current_word:
                   current_word = word_id
                   current_word_index += 1
               mapping[current_word_index].append(idx)


       # Randomly mask words
       mask = np.random.binomial(1, wwm_probability, (len(mapping),))
       input_ids = feature["input_ids"]
       labels = feature["labels"]
       new_labels = [-100] * len(labels)
       for word_id in np.where(mask)[0]:
           word_id = word_id.item()
           for idx in mapping[word_id]:
               new_labels[idx] = labels[idx]
               input_ids[idx] = tokenizer.mask_token_id
       feature["labels"] = new_labels


   return data_collator(features)




def save_model_checkpoint():
   pass




def load_model_checkpoint():
   pass


def make_output_dir(config: Dict):


   # paths to store model, checkpoints and plots
   output_dir = config.get("output_dir")
   #make_dir_path(output_dir)


   # store training config to file
   with open(os.path.join(output_dir, "train_config.json"), "w+") as output_config:
       output_config.write(json.dumps(config))




def select_model_and_tokenizer(model_name):
  
   if 'camembert' in model_name:
       tokenizer = CamembertTokenizer.from_pretrained(model_name)
       if 'causal' in model_name:  # causal LM is unidirectional, more suited for TextGen
           pretrained_model = CamembertForCausalLM.from_pretrained(
               model_name)
       else:
           pretrained_model = CamembertForMaskedLM.from_pretrained(
               model_name)
       return tokenizer, pretrained_model
  
   elif 'roberta' in model_name:
       tokenizer = RobertaTokenizer.from_pretrained(model_name)
       if 'causal' in model_name: 
           pretrained_model = RobertaForCausalLM.from_pretrained(
               model_name)
       else:
           pretrained_model = RobertaForMaskedLM.from_pretrained(
               model_name)
       return tokenizer, pretrained_model


   elif 'flaubert' in model_name:  # no causal LM for Flaubert ?
       tokenizer = FlaubertTokenizer.from_pretrained(model_name)
       pretrained_model = FlaubertWithLMHeadModel.from_pretrained(
           model_name)
       return tokenizer, pretrained_model


   tokenizer = RobertaTokenizer.from_pretrained(model_name)
   if 'causal' in model_name:
       pretrained_model = AutoModelForCausalLM.from_pretrained(
           model_name)
   else:
       pretrained_model = AutoModelForMaskedLM.from_pretrained(
           model_name)
   return tokenizer, pretrained_model



webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="models-pre-training")
def train():
   config= read_json_file("/home/raimbault/PRE/config.json")
   tokenizer, model = select_model_and_tokenizer("flaubert/flaubert_base_uncased")

   dataset = Dataset.from_generator(
       generator=generate_jsonl_lines,
       gen_kwargs={
           "root_path": "/home/raimbault/MLM",
           "recursive": True})


   tokenized_dataset = dataset.map(
       tokenize_function,
       batched=True,
       remove_columns=["text"],
       fn_kwargs={'tokenizer': tokenizer}
   )

   chunked_dataset = tokenized_dataset.map(
       group_texts,
       batched=True
   )

   chunked_dataset = chunked_dataset.train_test_split(train_size=0.95, test_size=0.05)

   if True :
        chunked_dataset['train'] = chunked_dataset['train'].select(range(0, 1000))
        chunked_dataset['test'] = chunked_dataset['test'].select(range(0, 100))

   data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer, mlm=True, mlm_probability=0.15
   )

   optimizer = AdamW(params=model.parameters(),lr=6e-4, betas=(0.9,0.98), eps=1e-6)
   lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer= optimizer, 
       num_warmup_steps=10000, num_training_steps=100000, power=config["power"], lr_end=config["lr_end"])

   training_args = TrainingArguments(
       output_dir=config["output_dir"],
       overwrite_output_dir=True,
       evaluation_strategy="steps",
       save_strategy=config["save_strategy"],
       eval_steps=config["eval_steps"],
       save_steps=config["save_steps"],
       logging_steps=config["logging_steps"],
       save_total_limit=config["save_total_limit"],
       learning_rate=config["learning_rate"],
       weight_decay=config["weight_decay"],
       #warmup_steps=4000,
       adam_beta2=config["adam_beta2"],
       adam_epsilon=config["adam_epsilon"],
       per_device_train_batch_size=config["batch_size"],
       per_device_eval_batch_size=config["batch_size"],
       push_to_hub=False,
       #lr_scheduler_type=scheduler,
       fp16=config["fp16"],  # set to true for training on CUDA
       remove_unused_columns=True,
       num_train_epochs=10,
       #num_train_epochs=config.get("num_train_epochs"),
       max_steps=config["max_steps"]
   )


   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=chunked_dataset['train'],
       eval_dataset=chunked_dataset['test'],
       data_collator=data_collator,
       tokenizer=tokenizer,
       optimizers=(optimizer,lr_scheduler),
       callbacks=[LearningRatePlotCallback,LossPlotCallback],

   )


   start = time.time()
   trainer.train()
   stop = time.time()
   logging.warning(
       "TOTAL TIME TAKEN FOR TRAINING : "
       f"{datetime.timedelta(seconds=stop-start)}"
       f" or {stop - start} seconds"
   )
   with open(os.path.join(config["output_dir"], "config.json"), "w") as configfile :
    configfile.write(json.dumps(config))

if __name__ == "__main__":
    train()


