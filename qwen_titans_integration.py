#!/usr/bin/env python3
"""
TITANS integration with Qwen 3 14B for extended context windows
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from titans_pytorch import NeuralMemory, MemoryAsContextTransformer
import argparse
import time
from typing import Optional, Dict, Any

class QwenTitansModel(nn.Module):
    """
    Qwen 3 14B model enhanced with TITANS neural memory for extended context windows
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        max_context_length: int = 32768,
        neural_memory_layers: tuple = (8, 16, 24),
        num_persist_mem_tokens: int = 8,
        num_longterm_mem_tokens: int = 32,
        neural_memory_segment_len: int = 8,
        neural_memory_batch_size: int = 256,
        use_flash_attention: bool = True,
        device: str = "auto"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.device = self._setup_device(device)
        
        print(f"Loading Qwen model: {model_name}")
        print(f"Target device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model config
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        # Extract model dimensions
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        print(f"Model config - Vocab: {self.vocab_size}, Hidden: {self.hidden_size}, Layers: {self.num_layers}")
        
        # Create TITANS memory-enhanced transformer
        self.titans_transformer = MemoryAsContextTransformer(
            num_tokens=self.vocab_size,
            dim=self.hidden_size,
            depth=self.num_layers,
            segment_len=512,  # Local attention window
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            neural_memory_layers=neural_memory_layers,
            neural_memory_segment_len=neural_memory_segment_len,
            neural_memory_batch_size=neural_memory_batch_size,
            use_flex_attn=use_flash_attention,
            sliding_window_attn=True,
            neural_memory_kwargs=dict(
                dim_head=64,
                heads=8,
                attn_pool_chunks=True,
                qk_rmsnorm=True,
                momentum=True,
                momentum_order=1,
                default_step_transform_max_lr=1e-2,
                use_accelerated_scan=True,
                per_parameter_lr_modulation=True,
                spectral_norm_surprises=True
            )
        )
        
        # Move to device
        if self.device.type != "cuda":  # Only move if not using device_map
            self.titans_transformer = self.titans_transformer.to(self.device)
        
        print(f"✓ TITANS integration initialized with {len(neural_memory_layers)} memory layers")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def generate_with_extended_context(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_titans: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with extended context using TITANS memory
        """
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        
        input_length = input_ids.shape[1]
        print(f"Input length: {input_length} tokens")
        
        if use_titans and input_length > 2048:
            print("Using TITANS for extended context generation...")
            generated_ids = self._generate_with_titans(
                input_ids, max_new_tokens, temperature, top_p, do_sample
            )
        else:
            print("Using standard generation...")
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        # Extract only the new generated part
        new_text = generated_text[len(prompt):]
        
        generation_time = time.time() - start_time
        
        return {
            "generated_text": new_text,
            "full_text": generated_text,
            "input_length": input_length,
            "output_length": generated_ids.shape[1] - input_length,
            "total_length": generated_ids.shape[1],
            "generation_time": generation_time,
            "tokens_per_second": (generated_ids.shape[1] - input_length) / generation_time
        }
    
    def _generate_with_titans(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> torch.Tensor:
        """Generate using TITANS memory-enhanced transformer"""
        
        with torch.no_grad():
            # Use TITANS transformer for generation
            generated_ids = self.titans_transformer.sample(
                input_ids,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_cache=True
            )
        
        return generated_ids
    
    def benchmark_context_lengths(
        self,
        base_prompt: str,
        context_lengths: list = [1024, 2048, 4096, 8192, 16384],
        max_new_tokens: int = 100
    ):
        """Benchmark performance across different context lengths"""
        
        print("\nBenchmarking context lengths...")
        print("=" * 60)
        
        results = []
        
        for length in context_lengths:
            # Create prompt of target length
            repeated_text = " ".join([base_prompt] * (length // len(base_prompt.split()) + 1))
            tokens = self.tokenizer.encode(repeated_text)
            truncated_tokens = tokens[:length]
            test_prompt = self.tokenizer.decode(truncated_tokens)
            
            print(f"\nTesting context length: {length} tokens")
            
            try:
                # Test with TITANS
                result_titans = self.generate_with_extended_context(
                    test_prompt,
                    max_new_tokens=max_new_tokens,
                    use_titans=True,
                    do_sample=False
                )
                
                # Test without TITANS (if context is small enough)
                if length <= 2048:
                    result_standard = self.generate_with_extended_context(
                        test_prompt,
                        max_new_tokens=max_new_tokens,
                        use_titans=False,
                        do_sample=False
                    )
                else:
                    result_standard = None
                
                results.append({
                    "context_length": length,
                    "titans_time": result_titans["generation_time"],
                    "titans_tps": result_titans["tokens_per_second"],
                    "standard_time": result_standard["generation_time"] if result_standard else None,
                    "standard_tps": result_standard["tokens_per_second"] if result_standard else None,
                    "success": True
                })
                
                print(f"✓ TITANS: {result_titans['tokens_per_second']:.2f} tokens/sec")
                if result_standard:
                    print(f"✓ Standard: {result_standard['tokens_per_second']:.2f} tokens/sec")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    "context_length": length,
                    "success": False,
                    "error": str(e)
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="TITANS + Qwen 3 14B Integration")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Qwen model name")
    parser.add_argument("--prompt", default="The future of artificial intelligence", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run context length benchmark")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    print("TITANS + Qwen 3 14B Integration")
    print("=" * 50)
    
    # Initialize model
    model = QwenTitansModel(
        model_name=args.model,
        device=args.device
    )
    
    if args.benchmark:
        # Run benchmark
        results = model.benchmark_context_lengths(args.prompt)
        
        print("\nBenchmark Results:")
        print("-" * 60)
        for result in results:
            if result["success"]:
                print(f"Context {result['context_length']:5d}: TITANS {result['titans_tps']:6.2f} tok/s")
            else:
                print(f"Context {result['context_length']:5d}: FAILED - {result['error']}")
    
    else:
        # Single generation
        print(f"\nGenerating with prompt: '{args.prompt}'")
        print("-" * 50)
        
        result = model.generate_with_extended_context(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\nGenerated text:")
        print(result["generated_text"])
        print(f"\nStats:")
        print(f"Input length: {result['input_length']} tokens")
        print(f"Output length: {result['output_length']} tokens")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print(f"Speed: {result['tokens_per_second']:.2f} tokens/sec")

if __name__ == "__main__":
    main()
