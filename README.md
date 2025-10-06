# Tech Challenge 3 — Fine-tuning do Llama 3.2 1B Instruct com LoRA (Unsloth)

Este repositório contém um notebook para realizar fine-tuning leve (LoRA) do modelo `unsloth/Llama-3.2-1B-Instruct` usando as bibliotecas Unsloth, TRL e PEFT. O objetivo é treinar o modelo para gerar descrições de produtos a partir de seus títulos, com base nos dados em `data/trn.json`.

## Visão geral
- Modelo base: `unsloth/Llama-3.2-1B-Instruct`
- Formato de dados: JSON Lines (um JSON por linha) em `data/trn.json`
- Mapeamento (instrução -> entrada -> saída):
  - instrução: `"DESCRIBE ABOUT THE PRODUCT."`
  - input: campo `title`
  - output: campo `content`
- Técnica: LoRA aplicada a um modelo quantizado em 4 bits (via Unsloth + PEFT)
- Arquivo principal: `fine_tune_llama3.ipynb`
- Saídas de treino: `outputs/llama-3.2-1b-lora`

O notebook prepara os dados, aplica o template de chat do modelo, treina com LoRA e mostra como carregar os adaptadores para geração (inferência).

## Dados (data/trn.json)
- Espera-se um arquivo no formato JSONL, onde cada linha é um objeto com os campos:
  - `title`: título do produto (string)
  - `content`: descrição do produto (string)
- O notebook descarta linhas sem `content` (para garantir que a saída/target seja não vazia).
- A instrução usada durante o treino é fixa: `DESCRIBE ABOUT THE PRODUCT.`

Exemplo de linha em `data/trn.json`:
```json
{"title": "Girls Ballet Tutu Neon Pink", "content": "Uma linda saia tutu rosa neon para ballet, leve e confortável..."}
```

## Modelo e tokenizador
O notebook carrega o modelo e tokenizador com Unsloth:
- `model_id = 'unsloth/Llama-3.2-1B-Instruct'`
- `max_seq_length = 2048`
- `load_in_4bit = True` (quantização 4-bit para economia de memória)
- `model.config.use_cache = False` durante o treino (importante para não conflitar com o gradiente)

O template de chat do tokenizador é usado para montar exemplos do tipo:
- Usuário: `DESCRIBE ABOUT THE PRODUCT.\nTitle: {title}`
- Assistente: `{content}`

## Configuração LoRA usada
A configuração de LoRA aplicada no notebook (via `FastLanguageModel.get_peft_model`) é:
- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']`

Os adaptadores LoRA são salvos em `outputs/llama-3.2-1b-lora` ao final do treino.

## Hiperparâmetros de treinamento (TrainingArguments)
No notebook, os principais argumentos de treino são definidos assim:
- `per_device_train_batch_size = 16` => Quantidade de amostras processadas por GPU/CPU a cada passo de treino. Impacta diretamente o uso de memória. 
- `gradient_accumulation_steps = 2` => Faz acumular gradientes por 2 passos antes de atualizar os pesos. Aumenta o “tamanho de lote efetivo” sem exigir mais memória de uma só vez.
- `num_train_epochs = 3` => Quantas passagens completas sobre o conjunto de treino (épocas) serão feitas. Se max_steps também for definido, ele tem precedência e pode encerrar o treino antes de completar as épocas.
- `learning_rate = 3e-5` => Taxa de aprendizado inicial (ou máxima, dependendo do scheduler). Para LoRA em LLMs, valores comuns: 1e-4 a 5e-5 a 1e-5. Muito alto → instabilidade/perda explode; muito baixo → treino lento/subótimo.
- `logging_steps = 1` => Frequência (em passos) para registrar métricas (loss etc.).
- `save_steps = 100` => Frequência (em passos) para salvar checkpoints.
- `max_steps = 200` => limita o número total de passos; útil para testes rápidos
- `warmup_steps = 10` => Passos iniciais com aumento gradual da learning_rate (de 0 até o valor base). Ajuda na estabilidade no começo do treino.
- `fp16 = False` => Ativa precisão mista em 16 bits (float16). Reduz memória e pode acelerar, mas requer suporte de hardware.
- `bf16 = True` => Ativa precisão mista em bfloat16. Geralmente mais estável que fp16 e bem suportada em GPUs Ampere+ (A100, RTX 30/40) e TPUs.
- `optim = 'paged_adamw_8bit'` => Otimizador AdamW em 8 bits (via bitsandbytes) com paginação, reduzindo uso de memória.
- `lr_scheduler_type = 'cosine'` => Agenda de taxa de aprendizado cosseno. Começa no pico (após warmup) e decai suavemente até próximo de 0.
- `output_dir = 'outputs/llama-3.2-1b-lora'` => Pasta onde checkpoints, adaptadores LoRA e logs são salvos.
- `seed = 42` => Semente de aleatoriedade para reprodutibilidade (tokenização, embaralhamento, inicializações etc.).

Outros pontos importantes no pipeline do notebook:
- Tokenização com `max_length = 1024` (para os exemplos de treino). As labels são iguais a `input_ids` (treino causal LM).
- `packing = True` no `SFTTrainer` para aproveitar melhor o contexto juntando múltiplos exemplos.
- Split treino/val automático (5% para teste quando há mais de 20 amostras).

## Como executar o notebook
1. Abra o arquivo `fine_tune_llama3.ipynb` em um ambiente com Jupyter/Colab/Vscode.
2. Instale as dependências (há uma célula no início do notebook):
   ```bash
   %pip -q install --upgrade "unsloth>=2024.08.08" "transformers>=4.43.3" "datasets>=2.20.0" "accelerate>=0.33.0" "peft>=0.11.1" "trl>=0.9.4" "sentencepiece>=0.2.0" "huggingface_hub>=0.24.6" "triton>=2.3.1"
   ```
3. Garanta que `data/trn.json` exista e esteja no formato esperado.
4. Execute as células na ordem. Ao final do treino, os adaptadores serão salvos em `outputs/llama-3.2-1b-lora`.

Observação sobre token da Hugging Face:
- O notebook usa um token Hugging Face para baixar o modelo. Recomenda-se substituir o valor hardcoded por uma variável de ambiente e ler com `os.getenv('HF_TOKEN')` ou inserir o token de forma segura no ambiente de execução.

## Inferência com os adaptadores LoRA
Após o treino, o notebook demonstra como carregar o modelo base e anexar os adaptadores LoRA:
- Carrega-se novamente o `unsloth/Llama-3.2-1B-Instruct` (4-bit, `use_cache=True`),
- Aplica-se `PeftModel.from_pretrained(base_model, 'outputs/llama-3.2-1b-lora')`.

Exemplo de uso (simplificado):
```python
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

model_id = 'unsloth/Llama-3.2-1B-Instruct'
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
base_model.config.use_cache = True
adapted = PeftModel.from_pretrained(base_model, 'outputs/llama-3.2-1b-lora')
adapted.eval()

def gerar_descricao(titulo: str, max_new_tokens: int = 128):
    messages = [
        { 'role': 'user', 'content': f'DESCRIBE ABOUT THE PRODUCT.\nTitle: {titulo}' }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(adapted.device)
    with torch.no_grad():
        out = adapted.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(gerar_descricao('Girls Ballet Tutu Neon Pink'))
```

## Dicas de desempenho/memória
- Se houver estouro de memória, reduza `max_length` da tokenização, aumente `gradient_accumulation_steps`, ou diminua `per_device_train_batch_size`.
- Manter `load_in_4bit=True` ajuda bastante a treinar em GPUs menores.
- Ajuste `max_steps`, `num_train_epochs` e `save_steps` conforme sua disponibilidade de tempo/recursos.

## Estrutura de diretórios relevante
- `data/trn.json`: dados de treino em JSONL.
- `fine_tune_llama3.ipynb`: notebook principal de fine-tuning.
- `outputs/llama-3.2-1b-lora`: diretório onde os adaptadores LoRA e estados de treino são salvos.
- `unsloth_compiled_cache`: cache compilado do Unsloth (pode ser recriado; geralmente não é necessário versionar).

## Licença
Consulte o arquivo `LICENSE` deste repositório e as licenças dos modelos e bibliotecas utilizadas (Unsloth, Transformers, Datasets, TRL, PEFT, etc.).