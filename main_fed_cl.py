from transformers import AutoTokenizer, BitsAndBytesConfig
import deepspeed
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig
)
import argparse
from loguru import logger
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
import copy
from component.collator import SFTDataCollator
from component.dataset import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer,ModifiedTrainer,ModifiedTrainer_EWC
from component.loss import TargetLMLoss


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/baichuan-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    print(training_args)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def local_update(idx,model,args,training_args,train_dataset,tokenizer,data_collator):
    training_args.num_train_epochs = 1
    if idx==0:
        training_args.output_dir = args.output_dir_fed1
        training_args.num_train_epochs = 1
    if idx==1:
        training_args.output_dir = args.output_dir_fed2
        training_args.num_train_epochs = 1
    if idx==2:
        training_args.output_dir = args.output_dir_fed3
        training_args.num_train_epochs = 1

    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset[idx],
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=None)
    model.save_pretrained(training_args.output_dir)

    return get_peft_model_state_dict(model)


def local_update_ewc(idx,model,args,training_args,train_dataset,tokenizer,data_collator,Importance_list,Star_vals):
    training_args.num_train_epochs = 1
    if idx==0:
        training_args.output_dir = args.output_dir_fed1
        training_args.num_train_epochs = 1
    if idx==1:
        training_args.output_dir = args.output_dir_fed2
        training_args.num_train_epochs = 1
    if idx==2:
        training_args.output_dir = args.output_dir_fed3
        training_args.num_train_epochs = 1

    trainer = ModifiedTrainer_EWC(
        model=model,
        args=training_args,
        train_dataset=train_dataset[idx],
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        others = [Importance_list[idx],Star_vals[idx]],
    )
    trainer.train(resume_from_checkpoint=None)
    model.save_pretrained(training_args.output_dir)

    return get_peft_model_state_dict(model)


def communication( server_model, w_locals, client_models,client_weights):
    client_num = len(w_locals)

    
    with torch.no_grad():
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            w_avg[k] = client_weights[0]*w_locals[0][k]

        for k in w_avg.keys():
            for i in range(1, client_num):  # i: 参与训练的 clients_num
                w_avg[k] += client_weights[i]*w_locals[i][k] # 各部分权重加和   

        ## 完成参数聚合
        set_peft_model_state_dict(server_model, w_avg)
        ## 分发更新后的参数
        w_globals = get_peft_model_state_dict(server_model)
        for client_idx in range(client_num):
            set_peft_model_state_dict(client_models[client_idx], w_globals)

    return server_model, client_models


def training(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    # 下面的设置至关重要，否则无法多卡训练
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    # 加载模型

    print("*** model loading ***")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False, 
        trust_remote_code=True,
        device_map="auto"
    )
    print("*** model finish ***")

    model.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()

     # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )


    # 部分tokenizer没有pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 如果两者相同，模型训练时不会计算eos_token_id的loss
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')


    # 初始化lora配置
    config1 = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["W_pack"],
        lora_dropout=args.lora_dropout
    )

    config2 = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["W_pack"],
        lora_dropout=args.lora_dropout
    )
    config3 = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["W_pack"],
        lora_dropout=args.lora_dropout
    )
    config4 = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["W_pack"],
        lora_dropout=args.lora_dropout
    )
    # 设置中心服务器
    server_model = get_peft_model(model, config1)
    server_model.print_trainable_parameters()
    server_model.config.torch_dtype = torch.float32

    # 设置本地服务器
    client_num = 3  # 默认是3个客户端，法考（考察对法学知识的掌握），法院数据，法律咨询数据
    client_weights = [0.25, 0.25, 0.5]  #  权重可提前计算，按照数据量
    # client_models = [copy.deepcopy(server_model) for idx in range(client_num)]
    client_models = [get_peft_model(model, config2),get_peft_model(model, config3),get_peft_model(model, config4)]


    # 加载训练集
    train_dataset = []
    train_dataset.append( SFTDataset(args.train_file_fed1, tokenizer, args.max_seq_length)  )
    train_dataset.append( SFTDataset(args.train_file_fed2, tokenizer, args.max_seq_length)  )
    train_dataset.append( SFTDataset(args.train_file_fed3, tokenizer, args.max_seq_length)  )
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    fed_all = training_args.output_dir
    print("*** starting training ***")
    for a_iter in range(args.fed_epochs):
        if a_iter == 0:
            w_locals = []
            wk_iters = 1
            for wi in range(wk_iters):
                print("============ Train epoch {} ============".format(wi + a_iter * wk_iters))
                for client_idx, model in enumerate(client_models):
                    local_w = local_update(client_idx,model,args,training_args,train_dataset,tokenizer,data_collator)
                    w_locals.append(copy.deepcopy(local_w))
            
            with torch.no_grad():
                server_model, client_models = communication( server_model, w_locals, client_models,client_weights)
                server_model.save_pretrained(fed_all)
        else:
            # 保存上一步的模型，不希望该模型在训练时遗忘太多内容
            Importance_list = [[],[],[]]  # 保存梯度
            Star_vals = [[],[],[]] # 保存参数
            for client_idx, model in enumerate(client_models):
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad==True:
                        Importance_list[client_idx].append(torch.abs(parameter.grad.data))
                        Star_vals[client_idx].append(parameter.data)
            w_locals = []
            wk_iters = 1
            for wi in range(wk_iters):
                print("============ Train epoch {} ============".format(wi + a_iter * wk_iters))
                for client_idx, model in enumerate(client_models):
                    local_w = local_update_ewc(client_idx,model,args,training_args,train_dataset,tokenizer,data_collator,Importance_list,Star_vals)
                    w_locals.append(copy.deepcopy(local_w))
            
            with torch.no_grad():
                server_model, client_models = communication( server_model, w_locals, client_models,client_weights)
                server_model.save_pretrained(fed_all)


    server_model.save_pretrained(fed_all)

    return server_model


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = training(args, training_args)



if __name__ == "__main__":
    main()

