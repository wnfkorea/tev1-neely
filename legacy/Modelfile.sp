FROM /home/wnf/tev1-finetune/tev1-neely-f16.gguf
TEMPLATE {{ .Prompt }}
PARAMETER num_ctx 4096
PARAMETER temperature 0
