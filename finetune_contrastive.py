import argparse
import os
import sys
from typing import List

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.trainer_callback import EarlyStoppingCallback

from utils import *
from collator import Collator
from modeling import create_contrastive_model, baseT5

def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    if args.load_model_name is not None:
        config = T5Config.from_pretrained(args.load_model_name)
        tokenizer = T5Tokenizer.from_pretrained(
            args.load_model_name,
            model_max_length=512,
        )
    else:
        config = T5Config.from_pretrained(args.base_model)
        tokenizer = T5Tokenizer.from_pretrained(
            args.base_model,
            model_max_length=512,
        )
        
    if args.tie_encoder_decoder:
        config.tie_encoder_decoder = True
        
    args.deepspeed = None
    gradient_checkpointing= False
    
    tasks = args.tasks.split(",")
    soft_prompts = {}
    for i, task in enumerate(tasks):
        if task == 'fgfusionseqrec':
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num * 4))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = []
            for k in range(4):
                p = ''.join(tokens[args.prompt_num * k : args.prompt_num * (k+1)])
                prompt.append(p)
            
        else:
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = ''.join(tokens)
            
        if local_rank == 0:
            print(tokens)
            
        soft_prompts[task] = prompt
    args.soft_prompts = soft_prompts

    train_data, valid_data = load_datasets(args)
    
    add_num = 0
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens())

    collator = Collator(args, tokenizer)
    
    if args.load_model_name is not None:
        model = create_contrastive_model(config)
        pretrained_model = T5ForConditionalGeneration.from_pretrained(args.load_model_name, config=config)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    else:
        print("load model from scratch: contrastive model")
        model = create_contrastive_model(config)

        
    model.resize_token_embeddings(len(tokenizer))

    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print('train sequence')
        for dataset in train_data.datasets:
            print(dataset[100])
        print('train sequence')
        print(valid_data[100])
        print(model)
        print_trainable_parameters(model)


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    early_stop = EarlyStoppingCallback(early_stopping_patience=args.patient)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            # bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=args.gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[early_stop]
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
