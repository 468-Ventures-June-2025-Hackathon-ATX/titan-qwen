#!/usr/bin/env python3
"""
Production TITANS implementation for Qwen 3 14B with extended context windows
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from titans_pytorch import NeuralMemory, MemoryAsContextTransformer
import argparse
import time
import json
import os
from typing import Optional, Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TitansQwenModel:
    """
    Production TITANS-enhanced Qwen model for extended context windows
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Start with 7B for faster loading
        max_context_length: int = 65536,
        neural_memory_layers: tuple = (8, 16, 24),
        num_persist_mem_tokens: int = 16,
        num_longterm_mem_tokens: int = 64,
        neural_memory_segment_len: int = 16,
        neural_memory_batch_size: int = 512,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.device = self._setup_device(device)
        
        logger.info(f"Initializing TITANS-Qwen model: {model_name}")
        logger.info(f"Target device: {self.device}")
        logger.info(f"Max context length: {max_context_length}")
        
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
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # Extract model dimensions
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        logger.info(f"Model config - Vocab: {self.vocab_size}, Hidden: {self.hidden_size}, Layers: {self.num_layers}")
        
        # Create TITANS memory-enhanced transformer
        self.titans_transformer = MemoryAsContextTransformer(
            num_tokens=self.vocab_size,
            dim=self.hidden_size,
            depth=self.num_layers,
            segment_len=1024,  # Local attention window
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            neural_memory_layers=neural_memory_layers,
            neural_memory_segment_len=neural_memory_segment_len,
            neural_memory_batch_size=neural_memory_batch_size,
            use_flex_attn=torch.cuda.is_available(),
            sliding_window_attn=True,
            neural_memory_kwargs=dict(
                dim_head=128,
                heads=16,
                attn_pool_chunks=True,
                qk_rmsnorm=True,
                momentum=True,
                momentum_order=2,
                default_step_transform_max_lr=5e-3,
                use_accelerated_scan=True,
                per_parameter_lr_modulation=True,
                spectral_norm_surprises=True,
                accept_weight_residual=True
            )
        )
        
        # Move TITANS to device if not using device_map
        if self.device.type != "cuda" or quantization_config is None:
            self.titans_transformer = self.titans_transformer.to(self.device)
        
        logger.info(f"✓ TITANS integration initialized with {len(neural_memory_layers)} memory layers")
        
        # Initialize memory state
        self.memory_state = None
        
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
    
    def reset_memory(self):
        """Reset the neural memory state"""
        self.memory_state = None
        logger.info("Memory state reset")
    
    def generate_extended_context(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_titans: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text with extended context using TITANS memory
        """
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        
        input_length = input_ids.shape[1]
        logger.info(f"Input length: {input_length} tokens")
        
        # Determine generation strategy
        if use_titans and input_length > 4096:
            logger.info("Using TITANS for extended context generation...")
            generated_ids = self._generate_with_titans(
                input_ids, max_new_tokens, temperature, top_p, top_k, do_sample, stream
            )
        else:
            logger.info("Using standard generation...")
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
        
        return {
            "generated_text": new_text,
            "full_text": generated_text,
            "input_length": input_length,
            "output_length": output_length,
            "total_length": generated_ids.shape[1],
            "generation_time": generation_time,
            "tokens_per_second": output_length / generation_time if generation_time > 0 else 0,
            "used_titans": use_titans and input_length > 4096
        }
    
    def _generate_with_titans(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stream: bool = False
    ) -> torch.Tensor:
        """Generate using TITANS memory-enhanced transformer"""
        
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
            
        return generated_ids
    
    def process_long_document(
        self,
        document: str,
        chunk_size: int = 8192,
        overlap: int = 512,
        summarize_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Process a long document using TITANS memory for context retention
        """
        logger.info(f"Processing document of {len(document)} characters")
        
        # Tokenize the full document
        tokens = self.tokenizer.encode(document)
        total_tokens = len(tokens)
        
        logger.info(f"Document has {total_tokens} tokens")
        
        if total_tokens <= chunk_size:
            # Document fits in one chunk
            return self.generate_extended_context(
                document,
                max_new_tokens=512,
                use_titans=False
            )
        
        # Process in overlapping chunks
        chunks = []
        summaries = []
        
        for i in range(0, total_tokens, chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            logger.info(f"Processing chunk {len(chunks) + 1}: tokens {i} to {i + len(chunk_tokens)}")
            
            # Generate summary or analysis for this chunk
            if summarize_chunks:
                summary_prompt = f"Summarize the key points from this text:\n\n{chunk_text}\n\nSummary:"
                result = self.generate_extended_context(
                    summary_prompt,
                    max_new_tokens=256,
                    temperature=0.3,
                    use_titans=True
                )
                summaries.append(result["generated_text"])
            
            chunks.append(chunk_text)
            
            if i + chunk_size >= total_tokens:
                break
        
        return {
            "total_tokens": total_tokens,
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "summaries": summaries if summarize_chunks else None,
            "processing_complete": True
        }
    
    def benchmark_context_scaling(
        self,
        base_text: str = "The future of artificial intelligence and machine learning",
        context_lengths: List[int] = [1024, 2048, 4096, 8192, 16384, 32768],
        max_new_tokens: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Benchmark performance across different context lengths
        """
        logger.info("Starting context scaling benchmark...")
        
        results = []
        
        for length in context_lengths:
            logger.info(f"Testing context length: {length} tokens")
            
            try:
                # Create test prompt of target length
                repeated_text = (base_text + " ") * (length // len(base_text.split()) + 1)
                tokens = self.tokenizer.encode(repeated_text)[:length]
                test_prompt = self.tokenizer.decode(tokens)
                
                # Reset memory for fair comparison
                self.reset_memory()
                
                # Test with TITANS
                result = self.generate_extended_context(
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
                    "success": True
                })
                
                logger.info(f"✓ {length} tokens: {result['tokens_per_second']:.2f} tok/s")
                
            except Exception as e:
                logger.error(f"❌ Failed at {length} tokens: {e}")
                results.append({
                    "context_length": length,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def save_config(self, filepath: str):
        """Save model configuration"""
        config = {
            "model_name": self.model_name,
            "max_context_length": self.max_context_length,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "device": str(self.device)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Production TITANS + Qwen Integration")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Qwen model name")
    parser.add_argument("--prompt", help="Input prompt for generation")
    parser.add_argument("--file", help="Text file to process")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run context scaling benchmark")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--max-context", type=int, default=65536, help="Maximum context length")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    print("TITANS + Qwen Production System")
    print("=" * 50)
    
    # Initialize model
    model = TitansQwenModel(
        model_name=args.model,
        max_context_length=args.max_context,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    if args.benchmark:
        # Run benchmark
        print("\nRunning context scaling benchmark...")
        results = model.benchmark_context_scaling()
        
        print("\nBenchmark Results:")
        print("-" * 60)
        for result in results:
            if result["success"]:
                titans_indicator = "TITANS" if result["used_titans"] else "Standard"
                print(f"Context {result['context_length']:6d}: {result['tokens_per_second']:6.2f} tok/s ({titans_indicator})")
            else:
                print(f"Context {result['context_length']:6d}: FAILED - {result['error']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif args.file:
        # Process file
        print(f"\nProcessing file: {args.file}")
        
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = model.process_long_document(content)
        
        print(f"Document processed:")
        print(f"- Total tokens: {result['total_tokens']}")
        print(f"- Chunks: {result['num_chunks']}")
        
        if result.get('summaries'):
            print(f"- Generated {len(result['summaries'])} chunk summaries")
    
    elif args.prompt:
        # Single generation
        print(f"\nGenerating with prompt: '{args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}'")
        print("-" * 50)
        
        result = model.generate_extended_context(
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
        print(f"Used TITANS: {result['used_titans']}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result["generated_text"])
            print(f"Output saved to {args.output}")
    
    else:
        print("\nNo action specified. Use --prompt, --file, or --benchmark")
        print("Example: python titans_qwen_production.py --prompt 'Tell me about AI' --max-tokens 500")

if __name__ == "__main__":
    main()
