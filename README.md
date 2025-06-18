# 🧠 Local Language Models in Therapy: A Semantic Similarity Evaluation

## Overview
This project explored the potential of **local large language models (LLMs)** as supportive tools in mental health care. With growing shortages of licensed therapists and long wait times, local LLMs offer a private, offline-accessible alternative that could help bridge the gap — especially in less severe cases or between therapy sessions.

## Objective
Evaluate how similar the responses of local LLMs are to those of real therapists using **semantic similarity** measures.

## Dataset
- **Source**: [mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- **Size**: 3,512 patient-therapist dialogue pairs

## Models Used
All models were used in **quantized (Q6) GGUF format** for efficient local deployment:
- [Qwen3-8B](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF)
- [LLaMA3-8B](https://huggingface.co/second-state/Llama-3-8B-Instruct-GGUF)
- [Mistral-7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [Gemma3-4B](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF)
It is reasonable to assume that these or at least comparable models would be used by the general public.

## Methodology
1. **Prompt Injection**: Feed patient statements as input to each model.
2. **Response Collection**: Store generated replies from each model.
3. **Embedding Generation**: Compute embeddings for model and therapist responses.
4. **Semantic Comparison**: Use **cosine similarity** to compare embeddings.

## Evaluation Metrics
- **Within-Human Similarity**: Baseline similarity between responses by different human therapists.
- **Model-Human Similarity**: Comparison of model responses with therapist responses.

## Results
- Considerable variance regarding the similarity between human responses to the same context
- All four evaluated local models perform similarly
- level and variance of local models similarity to human responses is comparable to within-human similarity for the same context
- model responses to some contexts showed low similarity to human response (needs further investigation, potentially simply due to short model answer)
  
## Conclusions
- Local LLMs show **promising potential** as tools to augment mental health services.
- Might act as **co-therapists** for mild conditions or interim support between therapy sessions in the future.
- Require **fine-tuning on therapeutic content** and **rigorous safety checks** before deployment.

## Potential Future Work
- safety and quality check regarding the human and model responses
- Fine-tune models on domain-specific **high-quality** (!!) therapeutic datasets
- Explore integration into hybrid therapy workflows

---

*This work contributes to ongoing discussions about safe, ethical, and effective AI-assisted therapy.*
