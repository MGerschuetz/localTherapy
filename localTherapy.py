import json
import os
import torch
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import csv
import tqdm as tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Parallelize therapy model evaluation across GPUs")
    parser.add_argument("--json_file", type=str, default="mental_health_counseling_conversations.json",
                        help="Path to the JSONL file with conversations")
    parser.add_argument("--output_csv", type=str, default="model_comparison_results",
                        help="Prefix for output CSV files")
    parser.add_argument("--embed_model", type=str, default="nomic-ai/nomic-embed-text-v2-moe",
                        help="Embedding model to use")
    parser.add_argument("--num_examples", type=int, default=-1,
                        help="Number of examples to process (use -1 for all)")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum tokens for model generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for model generation")
    parser.add_argument("--n_ctx", type=int, default=2048,
                        help="Context window size")
    parser.add_argument("--n_threads", type=int, default=4,
                        help="Number of CPU threads to use")
    return parser.parse_args()

libs_hf = [
    "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
    "second-state/Llama-3-8B-Instruct-GGUF",
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
]
files = [
    "DeepSeek-R1-0528-Qwen3-8B-UD-Q6_K_XL.gguf",
    "gemma-3-4b-it-UD-Q6_K_XL.gguf",
    "Meta-Llama-3-8B-Instruct-Q6_K.gguf",
    "mistral-7b-instruct-v0.2.Q6_K.gguf"
]

def setup_logging(gpu_idx):
    """Setup detailed logging for each GPU process"""
    logger = logging.getLogger(f'GPU_{gpu_idx}')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        f'[GPU {gpu_idx}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(f'gpu_{gpu_idx}_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def parse_jsonl_file(file_path):
    """
    Parse a JSONL file where each line is a JSON object with 'Context' and 'Response' fields.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of dictionaries, each containing 'Context' and 'Response' entries
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    # Extract Context and Response
                    context = json_obj.get('Context')
                    response = json_obj.get('Response')
                    
                    data.append({
                        'Context': context,
                        'Response': response
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error details: {e}")
    
    return data

def run_model_on_gpu(model_idx, gpu_idx, args):
    # Setup logging first
    logger = setup_logging(gpu_idx)
    logger.info("="*50)
    logger.info(f"STARTING GPU {gpu_idx} PROCESS FOR MODEL {model_idx}")
    logger.info("="*50)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # Restrict to one GPU per process
    logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_idx}")
    
    import torch
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import csv
    from sentence_transformers import SentenceTransformer, util
    
    # Set device
    logger.info("Setting CUDA device to 0 (relative to CUDA_VISIBLE_DEVICES)")
    torch.cuda.set_device(0)  # Since we're using CUDA_VISIBLE_DEVICES, this is always 0
    
    # Comprehensive GPU debugging
    logger.info("CUDA ENVIRONMENT CHECK:")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA Device: {torch.cuda.current_device()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"GPU Name: {torch.cuda.get_device_properties(0).name}")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        logger.error("CUDA NOT AVAILABLE!")
        return []

    def parse_jsonl_file(file_path):
        logger.info(f"Parsing JSONL file: {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get('Context')
                        response = json_obj.get('Response')
                        data.append({'Context': context, 'Response': response})
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {line}")
                        logger.error(f"Error details: {e}")
        logger.info(f"Parsed {len(data)} entries from JSONL file")
        return data

    logger.info("LOADING EMBEDDING MODEL:")
    logger.info(f"Model: {args.embed_model}")
    try:
        embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True, device="cuda")
        logger.info("Embedding model loaded successfully on GPU")
        logger.info(f"GPU Memory after embedding model - Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory after embedding model - Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return []

    def embed(text):
        return embed_model.encode(text, convert_to_tensor=True, device="cuda")

    logger.info("DOWNLOADING GGUF MODEL:")
    logger.info(f"Repository: {libs_hf[model_idx]}")
    logger.info(f"Filename: {files[model_idx]}")
    try:
        model_path = hf_hub_download(
            repo_id=libs_hf[model_idx],
            filename=files[model_idx]
        )
        logger.info(f"Model downloaded to: {model_path}")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return []
    
    # Enhanced GPU configuration for llama-cpp with detailed logging
    logger.info("INITIALIZING LLAMA MODEL:")
    logger.info(f"Model path: {model_path}")
    logger.info(f"n_ctx: {args.n_ctx}")
    logger.info(f"n_threads: {args.n_threads}")
    logger.info(f"n_gpu_layers: -1 (all layers)")
    logger.info(f"offload_kqv: True")
    logger.info(f"flash_attn: True")
    logger.info(f"use_mmap: True")
    logger.info(f"use_mlock: False")
    logger.info(f"verbose: True")
    logger.info(f"n_batch: 512")
    
    try:
        logger.info("Creating Llama instance...")
        llm = Llama(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=-1,  # Load ALL layers on GPU
            use_mmap=True,    # Use memory mapping
            use_mlock=False,  # Don't lock memory
            verbose=True,     # Show detailed loading info
            n_batch=512,      # Batch size for prompt processing
            offload_kqv=True, # Offload KV cache to GPU
            flash_attn=True   # Use flash attention if available
        )
        logger.info("Llama model created successfully")
        
        # Check GPU memory after model loading
        if torch.cuda.is_available():
            logger.info(f"GPU Memory after Llama model - Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"GPU Memory after Llama model - Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
    except Exception as e:
        logger.error(f"Error loading Llama model: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return []

    results = []
    combined_results = []
    examples = parse_jsonl_file(args.json_file)
    
    # Determine how many examples to process
    num_to_process = len(examples) if args.num_examples == -1 else min(args.num_examples, len(examples))
    logger.info(f"Processing {num_to_process} examples")

    for j, item in enumerate(examples[:num_to_process]):
        context = item.get("Context", "").strip()
        reference = item.get("Response", "").strip()
        if not context or not reference:
            logger.warning(f"Skipping entry {j} - missing context or reference")
            continue
        logger.info(f"Processing entry {j}/{num_to_process}")
        logger.debug(f"Context length: {len(context)} chars")
        logger.debug(f"Reference length: {len(reference)} chars")
        
        prompt = f"{context}\n\nAnswer:"
        try:
            logger.debug("Generating response...")
            output = llm(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            generated = output["choices"][0]["text"].strip()
            logger.debug(f"Generated response length: {len(generated)} chars")
        except Exception as e:
            logger.error(f"Error generating text for entry {j}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            continue
            
        logger.debug("Computing embeddings...")
        emb_gen = embed(generated)
        emb_ref = embed(reference)
        similarity = util.cos_sim(emb_gen, emb_ref).item()
        logger.debug(f"Similarity score: {similarity:.4f}")
        
        scalar_ref = emb_ref.mean().item()
        scalar_gen = emb_gen.mean().item()
        scalar_diff = abs(scalar_ref - scalar_gen)
        results.append({
            "index": j,
            "context": context,
            "reference_response": reference,
            "generated_response": generated,
            "similarity": similarity,
            "embedding_reference_scalar": emb_ref.mean().item(),
            "embedding_generated_scalar": emb_gen.mean().item(),
            "embedding_scalar_diff": abs(emb_ref.mean().item() - emb_gen.mean().item()),
            "embedding_reference": emb_ref.tolist(),
            "embedding_generated": emb_gen.tolist()
        })
        combined_results.append({
            "model_name": libs_hf[model_idx],
            "index": j,
            "similarity": similarity,
            "embedding_reference_scalar": scalar_ref,
            "embedding_generated_scalar": scalar_gen,
            "embedding_scalar_diff": scalar_diff
        })
        
    logger.info("SAVING RESULTS:")
    df = pd.DataFrame(results)
    text_columns = ["context", "reference_response", "generated_response"]
    for col in text_columns:
        df[col] = df[col].str.replace("\n", " ", regex=False)
    df_light = df.drop(columns=["embedding_reference", "embedding_generated"])
    
    csv_filename = f"{args.output_csv}_{libs_hf[model_idx].replace('/','_')}.csv"
    json_filename = f"results_with_embeddings_{libs_hf[model_idx].replace('/','_')}.json"
    
    df_light.to_csv(csv_filename, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    df.to_json(json_filename, orient="records", indent=2)
    
    logger.info(f"CSV results saved to: {csv_filename}")
    logger.info(f"JSON results saved to: {json_filename}")
    logger.info("="*50)
    logger.info(f"COMPLETED GPU {gpu_idx} PROCESS FOR MODEL {model_idx}")
    logger.info("="*50)
    
    return combined_results

if __name__ == "__main__":
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Using {num_gpus} GPUs")
    print(f"📁 Processing file: {args.json_file}")
    print(f"📊 Processing {args.num_examples if args.num_examples != -1 else 'all'} examples")
    
    all_combined_results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i in range(len(libs_hf)):
            gpu_idx = i % num_gpus
            futures.append(executor.submit(run_model_on_gpu, i, gpu_idx, args))
        for f in futures:
            all_combined_results.extend(f.result())
    df_combined = pd.DataFrame(all_combined_results)
    df_combined.to_csv("combined_model_comparison.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("✅ Combined comparison saved to: combined_model_comparison.csv")
