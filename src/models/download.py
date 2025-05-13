from huggingface_hub import hf_hub_download

# hf_hub_download(repo_id="RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf", filename="Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf", local_dir=".")
# hf_hub_download(repo_id="RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf", filename="Qwen1.5-MoE-A2.7B-Chat.Q4_1.gguf", local_dir=".")
# hf_hub_download(repo_id="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF", filename="L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q4_0_4_4.gguf", local_dir=".")
# hf_hub_download(repo_id="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF", filename="L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q8_0.gguf", local_dir=".")
# hf_hub_download(repo_id="DavidAU/Mistral-MOE-4X7B-Dark-MultiVerse-Uncensored-Enhanced32-24B-gguf", 
#                 filename="M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-D_AU-IQ4_XS.gguf", local_dir=".")
# hf_hub_download(repo_id="DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF", 
#                 filename="L3.2-8X4B-MOE-V2-Dark-Champion-Inst-21B-uncen-ablit-D_AU-IQ4_XS.gguf", local_dir=".")
# hf_hub_download(repo_id="DavidAU/DeepSeek-MOE-4X8B-R1-Distill-Llama-3.1-Deep-Thinker-Uncensored-24B-GGUF", 
#                 filename="DS4X8R1L3.1-Dp-Thnkr-UnC-24B-D_AU-IQ4_XS.gguf", local_dir=".")
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat", 
                filename="mistral-7b-instruct-v0.1.Q4_K_S.gguf", local_dir=".")
# M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-D_AU-IQ4_XS.gguf

# DavidAU/DeepSeek-MOE-4X8B-R1-Distill-Llama-3.1-Deep-Thinker-Uncensored-24B-GGUF