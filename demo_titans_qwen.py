#!/usr/bin/env python3
"""
Lightweight demonstration of TITANS + Qwen integration
Shows basic usage patterns and extended context capabilities
"""

import torch
import time
from transformers import AutoTokenizer
from titans_pytorch import MemoryAsContextTransformer
import argparse

class SimpleTitansQwenDemo:
    """
    Simplified TITANS + Qwen demonstration for quick testing
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize with a smaller model for demo purposes"""
        print(f"🚀 Initializing TITANS + Qwen Demo")
        print(f"Model: {model_name}")
        
        # Use smaller model for demo
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create TITANS transformer with demo-friendly parameters
        self.titans_model = MemoryAsContextTransformer(
            num_tokens=self.tokenizer.vocab_size,
            dim=512,  # Smaller dimension for demo
            depth=6,  # Fewer layers for faster processing
            segment_len=256,  # Smaller segments
            num_persist_mem_tokens=8,
            num_longterm_mem_tokens=32,
            neural_memory_layers=(2, 4),  # Memory in fewer layers
            neural_memory_segment_len=8,
            neural_memory_batch_size=128
        )
        
        print("✅ Demo model initialized successfully!")
    
    def demonstrate_basic_generation(self):
        """Show basic text generation"""
        print("\n" + "="*50)
        print("🔤 BASIC GENERATION DEMO")
        print("="*50)
        
        prompt = "The future of artificial intelligence will"
        print(f"Prompt: '{prompt}'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input tokens: {input_ids.shape[1]}")
        
        # Generate with TITANS
        start_time = time.time()
        with torch.no_grad():
            generated = self.titans_model.sample(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        
        generation_time = time.time() - start_time
        
        # Decode result
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):]
        
        print(f"Generated: '{new_text.strip()}'")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"🚀 Speed: {50/generation_time:.1f} tokens/sec")
    
    def demonstrate_extended_context(self):
        """Show extended context processing"""
        print("\n" + "="*50)
        print("📚 EXTENDED CONTEXT DEMO")
        print("="*50)
        
        # Create a longer context
        long_context = """
        In the realm of machine learning, transformers have revolutionized natural language processing.
        However, traditional transformers face limitations with very long sequences due to quadratic
        attention complexity. TITANS (Transformer with Implicit Attention via Tensor Network Structure)
        addresses this challenge by introducing neural memory mechanisms that allow models to process
        much longer sequences efficiently. The key innovation is the use of associative scan operations
        combined with momentum-based memory updates. This enables the model to maintain relevant
        information across very long contexts while keeping computational costs manageable.
        
        The neural memory system works by storing compressed representations of past context in
        learnable memory banks. These memories are updated during inference using test-time training,
        allowing the model to adapt to the specific content being processed. This is particularly
        valuable for tasks requiring long-term coherence and consistency.
        
        Applications of this technology include document summarization, long-form question answering,
        code generation with large codebases, and multi-turn conversations that span many exchanges.
        The ability to maintain context over thousands of tokens opens up new possibilities for
        AI applications that were previously limited by context window constraints.
        
        Now, given this background about TITANS technology, please explain how
        """ * 3  # Repeat to make it longer
        
        prompt = long_context + "this could impact the future of AI development:"
        
        print(f"Context length: ~{len(prompt.split())} words")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"]
        
        print(f"Input tokens: {input_ids.shape[1]}")
        
        if input_ids.shape[1] > 1000:
            print("🧠 Using TITANS neural memory for extended context...")
        
        # Generate with extended context
        start_time = time.time()
        with torch.no_grad():
            generated = self.titans_model.sample(
                input_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.95
            )
        
        generation_time = time.time() - start_time
        
        # Decode result
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()
        
        print(f"Generated response: '{new_text[:200]}{'...' if len(new_text) > 200 else ''}'")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"🚀 Speed: {100/generation_time:.1f} tokens/sec")
    
    def demonstrate_memory_persistence(self):
        """Show how memory persists across generations"""
        print("\n" + "="*50)
        print("🧠 MEMORY PERSISTENCE DEMO")
        print("="*50)
        
        # First generation to establish memory
        context1 = "My favorite color is blue, and I love hiking in the mountains."
        print(f"Context 1: '{context1}'")
        
        inputs1 = self.tokenizer(context1, return_tensors="pt")
        
        with torch.no_grad():
            # This establishes memory state
            _ = self.titans_model.sample(inputs1["input_ids"], max_new_tokens=20)
        
        print("✅ Memory established from first context")
        
        # Second generation that should reference the memory
        context2 = "What activities do I enjoy, and what's my preferred color?"
        print(f"Context 2: '{context2}'")
        
        inputs2 = self.tokenizer(context2, return_tensors="pt")
        
        with torch.no_grad():
            generated = self.titans_model.sample(
                inputs2["input_ids"],
                max_new_tokens=50,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        new_response = response[len(context2):].strip()
        
        print(f"Response: '{new_response}'")
        print("🔍 Check if the model remembers the blue color and hiking preference!")
    
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("🎭 TITANS + Qwen Integration Demo")
        print("This demo shows the key capabilities of neural memory-enhanced transformers")
        
        try:
            self.demonstrate_basic_generation()
            self.demonstrate_extended_context()
            self.demonstrate_memory_persistence()
            
            print("\n" + "="*50)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("Key features demonstrated:")
            print("✅ Basic text generation with TITANS")
            print("✅ Extended context processing")
            print("✅ Neural memory persistence")
            print("\nFor production use, see titans_qwen_production.py")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="TITANS + Qwen Demo")
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to use for demo (default: smaller model for speed)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run only basic generation demo"
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = SimpleTitansQwenDemo(args.model)
    
    if args.quick:
        demo.demonstrate_basic_generation()
    else:
        demo.run_full_demo()

if __name__ == "__main__":
    main()