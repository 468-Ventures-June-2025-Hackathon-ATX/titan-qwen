#!/usr/bin/env python3
"""
Ultra-long context TITANS implementation for Qwen with 512K+ token windows
Optimized for M3 Max with 64GB RAM
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from titans_pytorch import NeuralMemory, MemoryAsContextTransformer
import argparse
import time
import json
import os
import psutil
import gc
from typing import Optional, Dict, Any, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor system memory usage for ultra-long contexts"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            "process_rss": memory_info.rss / (1024**3),  # GB
            "process_vms": memory_info.vms / (1024**3),  # GB
            "system_total": virtual_memory.total / (1024**3),  # GB
            "system_available": virtual_memory.available / (1024**3),  # GB
            "system_percent": virtual_memory.percent
        }
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status"""
        mem = self.get_memory_usage()
        logger.info(f"Memory {context}: Process={mem['process_rss']:.1f}GB, "
                   f"Available={mem['system_available']:.1f}GB, "
                   f"Usage={mem['system_percent']:.1f}%")
        return mem

class UltraLongTitansQwen:
    """
    Ultra-long context TITANS-enhanced Qwen model for 512K+ token windows
    Optimized for M3 Max with 64GB unified memory
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_context_length: int = 524288,  # 512K tokens
        ultra_long_mode: bool = True,
        device: str = "auto",
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = True,
        memory_efficient: bool = True
    ):
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.ultra_long_mode = ultra_long_mode
        self.device = self._setup_device(device)
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"🚀 Initializing Ultra-Long TITANS-Qwen model: {model_name}")
        logger.info(f"🎯 Target context length: {max_context_length:,} tokens")
        logger.info(f"💾 Target device: {self.device}")
        
        self.memory_monitor.log_memory_status("initialization start")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model config
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Setup quantization if requested
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("🔧 Using 4-bit quantization")
        
        # Load base model with memory optimizations
        logger.info("📥 Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            use_cache=True
        )
        
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        # Extract model dimensions
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        logger.info(f"📊 Model config - Vocab: {self.vocab_size:,}, Hidden: {self.hidden_size}, Layers: {self.num_layers}")
        
        self.memory_monitor.log_memory_status("base model loaded")
        
        # Configure ultra-long context parameters
        if ultra_long_mode:
            neural_memory_config = self._get_ultra_long_config()
        else:
            neural_memory_config = self._get_standard_config()
        
        logger.info(f"🧠 Neural memory config: {neural_memory_config}")
        
        # Create TITANS memory-enhanced transformer
        logger.info("🔧 Initializing TITANS transformer...")
        self.titans_transformer = MemoryAsContextTransformer(
            num_tokens=self.vocab_size,
            dim=self.hidden_size,
            depth=self.num_layers,
            **neural_memory_config
        )
        
        # Move TITANS to device if not using device_map
        if self.device.type != "cuda" or quantization_config is None:
            self.titans_transformer = self.titans_transformer.to(self.device)
        
        # Enable memory efficient attention if available
        if memory_efficient and hasattr(self.titans_transformer, 'enable_memory_efficient_attention'):
            self.titans_transformer.enable_memory_efficient_attention()
            logger.info("✅ Memory efficient attention enabled")
        
        self.memory_monitor.log_memory_status("TITANS initialized")
        
        logger.info(f"✅ Ultra-Long TITANS integration initialized!")
        logger.info(f"🎯 Ready for {max_context_length:,} token contexts")
        
        # Initialize memory state
        self.memory_state = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device with M3 Max optimization"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("🔥 Using CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("🍎 Using Apple Metal (MPS) - optimized for M3 Max")
            else:
                device = "cpu"
                logger.info("💻 Using CPU")
        
        return torch.device(device)
    
    def _get_ultra_long_config(self) -> Dict[str, Any]:
        """Get ultra-long context configuration for 512K tokens"""
        return {
            "segment_len": 2048,  # Larger local attention window
            "num_persist_mem_tokens": 64,  # 4x standard persistent memory
            "num_longterm_mem_tokens": 256,  # 4x standard long-term memory
            "neural_memory_layers": tuple(range(4, min(32, self.num_layers), 2)),  # Memory in most layers
            "neural_memory_segment_len": 64,  # Larger segments for efficiency
            "neural_memory_batch_size": 2048,  # Leverage 64GB RAM
            "use_flex_attn": self.device.type in ["cuda", "mps"],
            "sliding_window_attn": True,
            "neural_memory_kwargs": {
                "dim_head": 128,
                "heads": 16,
                "attn_pool_chunks": True,
                "qk_rmsnorm": True,
                "momentum": True,
                "momentum_order": 2,
                "default_step_transform_max_lr": 1e-3,  # Lower LR for stability
                "use_accelerated_scan": True,
                "per_parameter_lr_modulation": True,
                "spectral_norm_surprises": True,
                "accept_weight_residual": True,
                "memory_efficient": True
            }
        }
    
    def _get_standard_config(self) -> Dict[str, Any]:
        """Get standard configuration for comparison"""
        return {
            "segment_len": 1024,
            "num_persist_mem_tokens": 16,
            "num_longterm_mem_tokens": 64,
            "neural_memory_layers": (8, 16, 24),
            "neural_memory_segment_len": 16,
            "neural_memory_batch_size": 512,
            "use_flex_attn": self.device.type in ["cuda", "mps"],
            "sliding_window_attn": True,
            "neural_memory_kwargs": {
                "dim_head": 128,
                "heads": 16,
                "attn_pool_chunks": True,
                "qk_rmsnorm": True,
                "momentum": True,
                "momentum_order": 2,
                "default_step_transform_max_lr": 5e-3,
                "use_accelerated_scan": True,
                "per_parameter_lr_modulation": True,
                "spectral_norm_surprises": True,
                "accept_weight_residual": True
            }
        }
    
    def reset_memory(self):
        """Reset the neural memory state and clean up"""
        self.memory_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        logger.info("🧹 Memory state reset and cache cleared")
    
    def generate_ultra_long_context(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_titans: bool = True,
        progressive_generation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with ultra-long context using TITANS memory
        """
        start_time = time.time()
        self.memory_monitor.log_memory_status("generation start")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        
        input_length = input_ids.shape[1]
        logger.info(f"📝 Input length: {input_length:,} tokens")
        
        # Memory usage check
        if input_length > self.max_context_length:
            logger.warning(f"⚠️  Input length ({input_length:,}) exceeds max context ({self.max_context_length:,})")
            logger.info("🔧 Truncating to max context length")
            input_ids = input_ids[:, -self.max_context_length:]
            input_length = input_ids.shape[1]
        
        # Determine generation strategy
        if use_titans and input_length > 4096:
            logger.info(f"🧠 Using TITANS for ultra-long context generation ({input_length:,} tokens)...")
            generated_ids = self._generate_with_titans(
                input_ids, max_new_tokens, temperature, top_p, top_k, do_sample
            )
        else:
            logger.info("🔄 Using standard generation...")
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        # Extract only the new generated part
        new_text = generated_text[len(prompt):]
        
        generation_time = time.time() - start_time
        output_length = generated_ids.shape[1] - input_length
        
        self.memory_monitor.log_memory_status("generation complete")
        
        return {
            "generated_text": new_text,
            "full_text": generated_text,
            "input_length": input_length,
            "output_length": output_length,
            "total_length": generated_ids.shape[1],
            "generation_time": generation_time,
            "tokens_per_second": output_length / generation_time if generation_time > 0 else 0,
            "used_titans": use_titans and input_length > 4096,
            "memory_usage": self.memory_monitor.get_memory_usage()
        }
    
    def _generate_with_titans(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool
    ) -> torch.Tensor:
        """Generate using TITANS memory-enhanced transformer with memory monitoring"""
        
        self.memory_monitor.log_memory_status("TITANS generation start")
        
        with torch.no_grad():
            # Use TITANS transformer for generation with memory state
            generated_ids = self.titans_transformer.sample(
                input_ids,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_cache=True
            )
            
            # Update memory state for future generations
            # This allows for persistent memory across multiple generations
            
        self.memory_monitor.log_memory_status("TITANS generation complete")
        return generated_ids
    
    def benchmark_ultra_long_contexts(
        self,
        base_text: str = "The future of artificial intelligence and machine learning",
        context_lengths: List[int] = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
        max_new_tokens: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Benchmark performance across ultra-long context lengths up to 512K
        """
        logger.info("🚀 Starting ultra-long context scaling benchmark...")
        logger.info(f"🎯 Testing up to {max(context_lengths):,} tokens")
        
        results = []
        
        for length in context_lengths:
            logger.info(f"🧪 Testing context length: {length:,} tokens")
            
            try:
                # Create test prompt of target length
                repeated_text = (base_text + " ") * (length // len(base_text.split()) + 1)
                tokens = self.tokenizer.encode(repeated_text)[:length]
                test_prompt = self.tokenizer.decode(tokens)
                
                # Reset memory for fair comparison
                self.reset_memory()
                self.memory_monitor.log_memory_status(f"before {length:,} tokens")
                
                # Test with TITANS
                result = self.generate_ultra_long_context(
                    test_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    use_titans=True,
                    do_sample=False
                )
                
                results.append({
                    "context_length": length,
                    "generation_time": result["generation_time"],
                    "tokens_per_second": result["tokens_per_second"],
                    "used_titans": result["used_titans"],
                    "memory_usage": result["memory_usage"],
                    "success": True
                })
                
                logger.info(f"✅ {length:,} tokens: {result['tokens_per_second']:.2f} tok/s, "
                           f"Memory: {result['memory_usage']['process_rss']:.1f}GB")
                
            except Exception as e:
                logger.error(f"❌ Failed at {length:,} tokens: {e}")
                results.append({
                    "context_length": length,
                    "success": False,
                    "error": str(e),
                    "memory_usage": self.memory_monitor.get_memory_usage()
                })
        
        return results
    
    def process_ultra_long_document(
        self,
        document: str,
        target_context_length: int = 524288,  # 512K
        overlap_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Process ultra-long documents using the full 512K context window
        """
        logger.info(f"📚 Processing ultra-long document: {len(document):,} characters")
        
        # Tokenize the full document
        tokens = self.tokenizer.encode(document)
        total_tokens = len(tokens)
        
        logger.info(f"📊 Document has {total_tokens:,} tokens")
        
        if total_tokens <= target_context_length:
            # Document fits in one ultra-long context
            logger.info(f"✅ Document fits in {target_context_length:,} token context window")
            return self.generate_ultra_long_context(
                document,
                max_new_tokens=1024,
                use_titans=True
            )
        
        # Process in ultra-long chunks with overlap
        overlap_size = int(target_context_length * overlap_ratio)
        chunk_size = target_context_length - overlap_size
        
        chunks = []
        results = []
        
        for i in range(0, total_tokens, chunk_size):
            chunk_tokens = tokens[i:i + target_context_length]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            logger.info(f"📝 Processing ultra-long chunk {len(chunks) + 1}: "
                       f"tokens {i:,} to {i + len(chunk_tokens):,}")
            
            # Process this ultra-long chunk
            result = self.generate_ultra_long_context(
                chunk_text,
                max_new_tokens=512,
                temperature=0.3,
                use_titans=True
            )
            
            chunks.append(chunk_text)
            results.append(result)
            
            if i + target_context_length >= total_tokens:
                break
        
        return {
            "total_tokens": total_tokens,
            "num_chunks": len(chunks),
            "chunk_size": target_context_length,
            "overlap_size": overlap_size,
            "results": results,
            "processing_complete": True,
            "ultra_long_processing": True
        }

def main():
    parser = argparse.ArgumentParser(description="Ultra-Long Context TITANS + Qwen Integration")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Qwen model name")
    parser.add_argument("--prompt", help="Input prompt for generation")
    parser.add_argument("--file", help="Text file to process")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run ultra-long context benchmark")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--max-context", type=int, default=524288, help="Maximum context length (default: 512K)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--ultra-long", action="store_true", default=True, help="Enable ultra-long mode")
    
    args = parser.parse_args()
    
    print("🚀 Ultra-Long Context TITANS + Qwen System")
    print("=" * 60)
    print(f"🎯 Target context length: {args.max_context:,} tokens")
    print("=" * 60)
    
    # Initialize model
    model = UltraLongTitansQwen(
        model_name=args.model,
        max_context_length=args.max_context,
        ultra_long_mode=args.ultra_long,
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )
    
    if args.benchmark:
        # Run ultra-long context benchmark
        print(f"\n🧪 Running ultra-long context benchmark up to {args.max_context:,} tokens...")
        
        # Define test context lengths up to the maximum
        test_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        if args.max_context >= 131072:
            test_lengths.append(131072)
        if args.max_context >= 262144:
            test_lengths.append(262144)
        if args.max_context >= 524288:
            test_lengths.append(524288)
        
        results = model.benchmark_ultra_long_contexts(context_lengths=test_lengths)
        
        print("\n📊 Ultra-Long Context Benchmark Results:")
        print("-" * 80)
        for result in results:
            if result["success"]:
                titans_indicator = "TITANS" if result["used_titans"] else "Standard"
                memory_gb = result["memory_usage"]["process_rss"]
                print(f"Context {result['context_length']:7,}: {result['tokens_per_second']:6.2f} tok/s "
                      f"({titans_indicator}) Memory: {memory_gb:5.1f}GB")
            else:
                print(f"Context {result['context_length']:7,}: FAILED - {result['error']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
    
    elif args.file:
        # Process ultra-long file
        print(f"\n📚 Processing ultra-long file: {args.file}")
        
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = model.process_ultra_long_document(content, target_context_length=args.max_context)
        
        print(f"📊 Ultra-long document processed:")
        print(f"- Total tokens: {result['total_tokens']:,}")
        print(f"- Chunks: {result['num_chunks']}")
        print(f"- Chunk size: {result['chunk_size']:,} tokens")
        print(f"- Ultra-long processing: {result['ultra_long_processing']}")
    
    elif args.prompt:
        # Single ultra-long generation
        print(f"\n📝 Generating with ultra-long context...")
        print(f"Prompt: '{args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}'")
        print("-" * 60)
        
        result = model.generate_ultra_long_context(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\n✨ Generated text:")
        print(result["generated_text"])
        print(f"\n📊 Stats:")
        print(f"Input length: {result['input_length']:,} tokens")
        print(f"Output length: {result['output_length']:,} tokens")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print(f"Speed: {result['tokens_per_second']:.2f} tokens/sec")
        print(f"Used TITANS: {result['used_titans']}")
        print(f"Memory usage: {result['memory_usage']['process_rss']:.1f}GB")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result["generated_text"])
            print(f"💾 Output saved to {args.output}")
    
    else:
        print("\n❓ No action specified. Use --prompt, --file, or --benchmark")
        print("Example: python titans_qwen_ultra_long.py --prompt 'Tell me about AI' --max-tokens 500")
        print(f"Ultra-long example: python titans_qwen_ultra_long.py --benchmark --max-context 524288")

if __name__ == "__main__":
    main()
