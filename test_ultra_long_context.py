#!/usr/bin/env python3
"""
Comprehensive test suite for ultra-long context TITANS implementation
Tests 512K token context windows on M3 Max with 64GB RAM
"""

import torch
import time
import json
import os
import sys
from typing import Dict, List, Any
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from titans_qwen_ultra_long import UltraLongTitansQwen, MemoryMonitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraLongContextTester:
    """
    Comprehensive test suite for ultra-long context functionality
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.memory_monitor = MemoryMonitor()
        self.test_results = []
        
        logger.info(f"🧪 Initializing Ultra-Long Context Test Suite")
        logger.info(f"📊 Model: {model_name}")
        
    def create_test_content(self, target_tokens: int) -> str:
        """Create test content of approximately target token length"""
        base_text = """
        Artificial intelligence represents one of the most transformative technologies of our time. 
        Machine learning algorithms have evolved from simple statistical models to complex neural networks 
        capable of understanding language, recognizing images, and making sophisticated decisions. 
        The development of transformer architectures has particularly revolutionized natural language processing, 
        enabling models to understand context and generate human-like text with remarkable fluency.
        
        Deep learning has opened new frontiers in computer vision, allowing machines to interpret 
        visual information with accuracy that often surpasses human capabilities. Convolutional neural 
        networks have become the backbone of image recognition systems, while recurrent neural networks 
        have excelled in processing sequential data like speech and text.
        
        The emergence of large language models has demonstrated the potential for artificial general 
        intelligence, though significant challenges remain. These models can engage in complex reasoning, 
        creative writing, and problem-solving across diverse domains. However, they also face limitations 
        in terms of factual accuracy, bias, and the ability to truly understand rather than pattern match.
        
        Reinforcement learning has shown promise in game-playing scenarios, robotics, and autonomous 
        systems. By learning through trial and error, these systems can develop strategies that sometimes 
        exceed human expert performance. The combination of deep learning with reinforcement learning 
        has led to breakthroughs in complex decision-making tasks.
        
        The future of AI likely involves the integration of multiple approaches: symbolic reasoning 
        combined with neural networks, improved architectures that can handle longer contexts and 
        maintain coherence over extended interactions, and systems that can learn more efficiently 
        from limited data while generalizing to new situations.
        """
        
        # Estimate tokens (roughly 4 characters per token)
        estimated_tokens_per_repeat = len(base_text) // 4
        repeats_needed = max(1, target_tokens // estimated_tokens_per_repeat)
        
        return (base_text.strip() + "\n\n") * repeats_needed
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic TITANS functionality with small context"""
        logger.info("🔧 Testing basic functionality...")
        
        try:
            model = UltraLongTitansQwen(
                model_name=self.model_name,
                max_context_length=8192,  # Small context for basic test
                ultra_long_mode=False
            )
            
            test_prompt = "The future of artificial intelligence will"
            result = model.generate_ultra_long_context(
                test_prompt,
                max_new_tokens=50,
                temperature=0.7
            )
            
            success = len(result["generated_text"]) > 0
            
            return {
                "test_name": "basic_functionality",
                "success": success,
                "input_length": result["input_length"],
                "output_length": result["output_length"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
                "memory_usage": result["memory_usage"]
            }
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            return {
                "test_name": "basic_functionality",
                "success": False,
                "error": str(e)
            }
    
    def test_progressive_context_scaling(self) -> List[Dict[str, Any]]:
        """Test progressive scaling from 1K to 512K tokens"""
        logger.info("📈 Testing progressive context scaling...")
        
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        results = []
        
        try:
            model = UltraLongTitansQwen(
                model_name=self.model_name,
                max_context_length=524288,
                ultra_long_mode=True
            )
            
            for length in context_lengths:
                logger.info(f"🧪 Testing {length:,} tokens...")
                
                try:
                    # Create test content
                    test_content = self.create_test_content(length)
                    
                    # Truncate to exact length
                    tokens = model.tokenizer.encode(test_content)[:length]
                    test_prompt = model.tokenizer.decode(tokens)
                    
                    # Reset memory for fair comparison
                    model.reset_memory()
                    
                    # Test generation
                    start_time = time.time()
                    result = model.generate_ultra_long_context(
                        test_prompt,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=False
                    )
                    
                    results.append({
                        "test_name": f"context_scaling_{length}",
                        "context_length": length,
                        "success": True,
                        "input_length": result["input_length"],
                        "output_length": result["output_length"],
                        "generation_time": result["generation_time"],
                        "tokens_per_second": result["tokens_per_second"],
                        "used_titans": result["used_titans"],
                        "memory_usage": result["memory_usage"]
                    })
                    
                    logger.info(f"✅ {length:,} tokens: {result['tokens_per_second']:.2f} tok/s, "
                               f"Memory: {result['memory_usage']['process_rss']:.1f}GB")
                    
                except Exception as e:
                    logger.error(f"❌ Failed at {length:,} tokens: {e}")
                    results.append({
                        "test_name": f"context_scaling_{length}",
                        "context_length": length,
                        "success": False,
                        "error": str(e)
                    })
            
        except Exception as e:
            logger.error(f"❌ Progressive scaling test setup failed: {e}")
            results.append({
                "test_name": "progressive_scaling_setup",
                "success": False,
                "error": str(e)
            })
        
        return results
    
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory efficiency with 512K context"""
        logger.info("💾 Testing memory efficiency with 512K context...")
        
        try:
            initial_memory = self.memory_monitor.get_memory_usage()
            
            model = UltraLongTitansQwen(
                model_name=self.model_name,
                max_context_length=524288,
                ultra_long_mode=True,
                memory_efficient=True
            )
            
            post_init_memory = self.memory_monitor.get_memory_usage()
            
            # Create 512K token content
            test_content = self.create_test_content(524288)
            tokens = model.tokenizer.encode(test_content)[:524288]
            test_prompt = model.tokenizer.decode(tokens)
            
            logger.info(f"📝 Created test content with {len(tokens):,} tokens")
            
            # Test generation with memory monitoring
            result = model.generate_ultra_long_context(
                test_prompt,
                max_new_tokens=256,
                temperature=0.7
            )
            
            final_memory = self.memory_monitor.get_memory_usage()
            
            memory_overhead = final_memory["process_rss"] - initial_memory["process_rss"]
            
            return {
                "test_name": "memory_efficiency_512k",
                "success": True,
                "context_length": 524288,
                "actual_input_length": result["input_length"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
                "initial_memory_gb": initial_memory["process_rss"],
                "post_init_memory_gb": post_init_memory["process_rss"],
                "final_memory_gb": final_memory["process_rss"],
                "memory_overhead_gb": memory_overhead,
                "system_memory_available_gb": final_memory["system_available"],
                "system_memory_usage_percent": final_memory["system_percent"]
            }
            
        except Exception as e:
            logger.error(f"❌ Memory efficiency test failed: {e}")
            return {
                "test_name": "memory_efficiency_512k",
                "success": False,
                "error": str(e)
            }
    
    def test_generation_quality(self) -> Dict[str, Any]:
        """Test generation quality with ultra-long context"""
        logger.info("✨ Testing generation quality with ultra-long context...")
        
        try:
            model = UltraLongTitansQwen(
                model_name=self.model_name,
                max_context_length=262144,  # 256K for quality test
                ultra_long_mode=True
            )
            
            # Create a coherent long context with specific information
            context = """
            Dr. Sarah Chen is a renowned AI researcher who has been working on neural memory systems 
            for the past decade. Her groundbreaking work on associative memory networks has led to 
            significant advances in long-term context retention for language models. She was born in 
            San Francisco in 1985 and completed her PhD at Stanford University in 2012.
            
            Her research team at the Institute for Advanced AI has developed several key innovations:
            1. Momentum-based memory updates that improve stability
            2. Spectral normalization techniques for surprise learning
            3. Hierarchical memory architectures that scale to millions of tokens
            
            Dr. Chen's favorite color is blue, and she enjoys hiking in the mountains during weekends. 
            She has published over 50 papers in top-tier conferences and has received the prestigious 
            Turing Award for her contributions to artificial intelligence.
            
            """ * 100  # Repeat to create long context
            
            # Add a question that requires remembering information from early in the context
            question = "\n\nBased on the information provided earlier, what is Dr. Sarah Chen's favorite color and what does she enjoy doing on weekends?"
            
            full_prompt = context + question
            
            result = model.generate_ultra_long_context(
                full_prompt,
                max_new_tokens=200,
                temperature=0.3  # Lower temperature for more focused answers
            )
            
            # Check if the answer contains the correct information
            generated_text = result["generated_text"].lower()
            contains_blue = "blue" in generated_text
            contains_hiking = "hiking" in generated_text or "mountains" in generated_text
            
            quality_score = (contains_blue + contains_hiking) / 2.0
            
            return {
                "test_name": "generation_quality",
                "success": True,
                "context_length": result["input_length"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
                "generated_text": result["generated_text"][:500],  # First 500 chars
                "contains_blue": contains_blue,
                "contains_hiking": contains_hiking,
                "quality_score": quality_score,
                "memory_usage": result["memory_usage"]
            }
            
        except Exception as e:
            logger.error(f"❌ Generation quality test failed: {e}")
            return {
                "test_name": "generation_quality",
                "success": False,
                "error": str(e)
            }
    
    def test_stability_over_time(self) -> Dict[str, Any]:
        """Test stability with multiple generations"""
        logger.info("⏱️ Testing stability over multiple generations...")
        
        try:
            model = UltraLongTitansQwen(
                model_name=self.model_name,
                max_context_length=131072,  # 128K for stability test
                ultra_long_mode=True
            )
            
            base_context = self.create_test_content(100000)  # ~100K tokens
            
            generation_times = []
            memory_usages = []
            
            for i in range(5):  # 5 consecutive generations
                logger.info(f"🔄 Generation {i+1}/5...")
                
                prompt = base_context + f"\n\nGeneration {i+1}: Please summarize the key points about AI development:"
                
                result = model.generate_ultra_long_context(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7
                )
                
                generation_times.append(result["generation_time"])
                memory_usages.append(result["memory_usage"]["process_rss"])
                
                logger.info(f"✅ Generation {i+1}: {result['tokens_per_second']:.2f} tok/s")
            
            # Calculate stability metrics
            avg_time = sum(generation_times) / len(generation_times)
            time_variance = sum((t - avg_time) ** 2 for t in generation_times) / len(generation_times)
            
            avg_memory = sum(memory_usages) / len(memory_usages)
            memory_variance = sum((m - avg_memory) ** 2 for m in memory_usages) / len(memory_usages)
            
            return {
                "test_name": "stability_over_time",
                "success": True,
                "num_generations": 5,
                "generation_times": generation_times,
                "memory_usages": memory_usages,
                "avg_generation_time": avg_time,
                "time_variance": time_variance,
                "avg_memory_usage": avg_memory,
                "memory_variance": memory_variance,
                "stable_performance": time_variance < (avg_time * 0.1) ** 2  # Less than 10% variance
            }
            
        except Exception as e:
            logger.error(f"❌ Stability test failed: {e}")
            return {
                "test_name": "stability_over_time",
                "success": False,
                "error": str(e)
            }
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🚀 Starting Comprehensive Ultra-Long Context Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        test_results = {
            "test_suite": "ultra_long_context_comprehensive",
            "model_name": self.model_name,
            "start_time": start_time,
            "system_info": {
                "total_memory_gb": self.memory_monitor.get_memory_usage()["system_total"],
                "available_memory_gb": self.memory_monitor.get_memory_usage()["system_available"],
                "device": "auto"
            },
            "tests": {}
        }
        
        # Test 1: Basic functionality
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Basic Functionality")
        logger.info("="*50)
        test_results["tests"]["basic_functionality"] = self.test_basic_functionality()
        
        # Test 2: Progressive context scaling
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Progressive Context Scaling")
        logger.info("="*50)
        test_results["tests"]["progressive_scaling"] = self.test_progressive_context_scaling()
        
        # Test 3: Memory efficiency
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Memory Efficiency (512K)")
        logger.info("="*50)
        test_results["tests"]["memory_efficiency"] = self.test_memory_efficiency()
        
        # Test 4: Generation quality
        logger.info("\n" + "="*50)
        logger.info("TEST 4: Generation Quality")
        logger.info("="*50)
        test_results["tests"]["generation_quality"] = self.test_generation_quality()
        
        # Test 5: Stability over time
        logger.info("\n" + "="*50)
        logger.info("TEST 5: Stability Over Time")
        logger.info("="*50)
        test_results["tests"]["stability"] = self.test_stability_over_time()
        
        total_time = time.time() - start_time
        test_results["total_time"] = total_time
        test_results["end_time"] = time.time()
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a comprehensive test summary"""
        logger.info("\n" + "="*80)
        logger.info("🎯 ULTRA-LONG CONTEXT TEST SUITE SUMMARY")
        logger.info("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_result in results["tests"].items():
            if isinstance(test_result, list):
                # Handle progressive scaling results
                for sub_result in test_result:
                    total_tests += 1
                    if sub_result.get("success", False):
                        passed_tests += 1
            else:
                total_tests += 1
                if test_result.get("success", False):
                    passed_tests += 1
        
        logger.info(f"📊 Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {total_tests - passed_tests}")
        logger.info(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"⏱️ Total Time: {results['total_time']:.2f}s")
        
        # Print key achievements
        logger.info("\n🏆 KEY ACHIEVEMENTS:")
        
        # Check for 512K success
        memory_test = results["tests"].get("memory_efficiency", {})
        if memory_test.get("success") and memory_test.get("context_length") == 524288:
            logger.info("✅ Successfully processed 512K token context")
            logger.info(f"   Memory usage: {memory_test.get('final_memory_gb', 0):.1f}GB")
            logger.info(f"   Speed: {memory_test.get('tokens_per_second', 0):.2f} tokens/sec")
        
        # Check for scaling performance
        scaling_results = results["tests"].get("progressive_scaling", [])
        if scaling_results:
            max_successful_context = 0
            for result in scaling_results:
                if result.get("success") and result.get("context_length", 0) > max_successful_context:
                    max_successful_context = result["context_length"]
            
            if max_successful_context > 0:
                logger.info(f"✅ Maximum successful context: {max_successful_context:,} tokens")
        
        # Check generation quality
        quality_test = results["tests"].get("generation_quality", {})
        if quality_test.get("success"):
            quality_score = quality_test.get("quality_score", 0)
            logger.info(f"✅ Generation quality score: {quality_score:.2f}/1.0")
        
        logger.info("\n" + "="*80)

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Long Context Test Suite")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to test")
    parser.add_argument("--output", help="Output file for test results")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = UltraLongContextTester(args.model)
    
    if args.quick:
        # Run only basic functionality test
        logger.info("🏃 Running quick test...")
        result = tester.test_basic_functionality()
        print(json.dumps(result, indent=2))
    else:
        # Run comprehensive test suite
        results = tester.run_comprehensive_test_suite()
        tester.print_test_summary(results)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Test results saved to {args.output}")

if __name__ == "__main__":
    main()
