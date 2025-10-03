#!/usr/bin/env python3
"""
超安全EIT-P训练脚本
进一步优化GPU内存使用，减少内存泄漏
"""

import torch
import gc
import os
import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments
from eit_p.training import EITPTrainer

def setup_environment():
    """设置环境变量以优化GPU内存使用"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用tokenizer并行
    
    # 设置PyTorch内存管理
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU内存状态:")
        print(f"  已分配: {allocated:.2f}GB")
        print(f"  已保留: {reserved:.2f}GB")
        print(f"  总内存: {total:.2f}GB")
        print(f"  使用率: {(allocated/total)*100:.1f}%")
        
        return allocated, reserved, total
    return 0, 0, 0

def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    print("=" * 60)
    print("EIT-P 超安全训练脚本")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 检查GPU状态
    print("\n1. 检查GPU状态...")
    allocated, reserved, total = check_gpu_memory()
    
    if total > 0 and allocated > total * 0.8:
        print("⚠️ 警告: GPU内存使用率过高，建议重启训练")
        return
    
    try:
        # 清理内存
        cleanup_memory()
        
        print("\n2. 加载模型...")
        model_name = 'gpt2'
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map="auto"  # 自动设备映射
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ 模型加载完成")
        check_gpu_memory()
        
        print("\n3. 准备数据集...")
        # 使用更小的block_size
        train_dataset = TextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=16)  # 进一步减小
        eval_dataset = TextDataset(tokenizer=tokenizer, file_path="eval.txt", block_size=16)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        print("✓ 数据集准备完成")
        
        print("\n4. 配置训练参数...")
        training_args = TrainingArguments(
            output_dir="./eitp_results",
            num_train_epochs=1,  # 只训练1个epoch
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # 增加梯度累积
            eval_strategy="no",  # 禁用评估以节省内存
            save_strategy="steps",
            save_steps=500,  # 减少保存频率
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=50,  # 减少日志频率
            fp16=False,
            dataloader_num_workers=0,
            max_grad_norm=0.1,
            warmup_steps=10,  # 减少warmup
            load_best_model_at_end=False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_drop_last=True,
            save_safetensors=True,
        )
        
        hypernetwork_params = {
            "input_dim": 2 + 2 * model.config.n_embd + 1,
            "hidden_dim": 8,  # 进一步减小
            "output_dim": 1,
            "num_layers": 1,
            "dropout": 0.5,  # 增加dropout
        }
        
        print("✓ 训练参数配置完成")
        
        print("\n5. 初始化训练器...")
        # 再次清理内存
        cleanup_memory()
        
        trainer = EITPTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            hypernetwork_params=hypernetwork_params,
        )
        
        print("✓ 训练器初始化完成")
        check_gpu_memory()
        
        print("\n6. 开始训练...")
        print("训练配置:")
        print(f"  Batch Size: {training_args.per_device_train_batch_size}")
        print(f"  Gradient Accumulation: {training_args.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"  Block Size: 16")
        print(f"  Hypernetwork Hidden Dim: {hypernetwork_params['hidden_dim']}")
        print(f"  Max Grad Norm: {training_args.max_grad_norm}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print("=" * 60)
        
        # 开始训练
        trainer.train()
        
        print("\n✓ 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        try:
            trainer.save_model("./eitp_results/interrupted_checkpoint")
            print("✓ 已保存中断检查点")
        except:
            print("✗ 保存检查点失败")
            
    except Exception as e:
        print(f"\n✗ 训练出错: {str(e)}")
        print("\n错误详情:")
        traceback.print_exc()
        
    finally:
        print("\n7. 清理资源...")
        cleanup_memory()
        check_gpu_memory()
        print("✓ 资源清理完成")

if __name__ == "__main__":
    main()
