
import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments
from eit_p.training import EITPTrainer

def main():
    # 设置环境变量以优化内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA操作以便调试
    
    model_name = 'gpt2'  # 使用小型模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        output_hidden_states=True,
        # torch_dtype=torch.float32,  # 使用FP32保证稳定性
        low_cpu_mem_usage=True  # 低内存模式加载
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 减小block_size以降低内存占用
    train_dataset = TextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=64)
    eval_dataset = TextDataset(tokenizer=tokenizer, file_path="eval.txt", block_size=64)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./eitp_results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 保持最小batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 增加梯度累积以补偿小batch size
        eval_strategy="steps",  # 改为steps以便更频繁检查点
        eval_steps=200,  # 更频繁的评估
        save_strategy="steps",
        save_steps=200,  # 更频繁的保存
        save_total_limit=1,  # 只保留最新1个checkpoint
        logging_dir="./logs",
        logging_steps=25,  # 更频繁的日志记录
        fp16=False,  # 禁用混合精度训练（chaos.py中的复数运算不支持FP16）
        dataloader_num_workers=0,  # 减少worker避免内存溢出
        max_grad_norm=0.3,  # 更严格的梯度裁剪
        warmup_steps=50,  # 减少warmup步数
        load_best_model_at_end=False,  # 避免加载额外模型
        report_to="none",  # 禁用wandb等减少开销
        remove_unused_columns=False,  # 避免删除列时的内存问题
        dataloader_pin_memory=False,  # 禁用pin_memory以节省内存
        dataloader_drop_last=True,  # 丢弃最后一个不完整的batch
    )

    hypernetwork_params = {
        "input_dim": 2 + 2 * model.config.n_embd + 1,  # lyapunov(1) + entropy(1) + h_mean(768) + h_std(768) + h_norm_mean(1) = 1539
        "hidden_dim": 32,  # 进一步减小hidden_dim到32
        "output_dim": 1,
        "num_layers": 2,  # 减少层数
        "dropout": 0.2,  # 增加dropout防止过拟合
    }

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    trainer = EITPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        hypernetwork_params=hypernetwork_params,
    )

    print("=" * 50)
    print("EIT-P Training Configuration (优化版):")
    print(f"  Batch Size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Precision: FP32 (稳定模式)")
    print(f"  Block Size: 64 (序列长度)")
    print(f"  Hypernetwork Hidden Dim: {hypernetwork_params['hidden_dim']}")
    print(f"  Save Strategy: 每{training_args.save_steps}步")
    print(f"  Max Checkpoints: {training_args.save_total_limit}")
    print("  优化特性: 小batch + 梯度累积 + 内存监控")
    print("=" * 50)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存checkpoint...")
        trainer.save_model("./eitp_results/interrupted_checkpoint")
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
