# Tech Challenge 3 — Fine-tuning com Unsloth (pt-BR)

Este repositório contém um notebook (tc3_fine_tuning.ipynb) que realiza fine-tuning leve (LoRA) de um modelo de linguagem a partir de uma lista de produtos. O objetivo é treinar o modelo para responder/gerar descrições coerentes dos produtos usando títulos e conteúdos fornecidos em um arquivo JSONL.

Resumo do fluxo (conforme o notebook):
- Ambiente alvo: Google Colab (com GPU, idealmente L4/A100), usando Google Drive para ler/gravar arquivos.
- Dados de entrada: JSON Lines (um JSON por linha) com campos `title` e `content`.
- Modelo base (pré-treinado): `unsloth/llama-3-8b-bnb-4bit` (carregado em 4 bits).
- Técnica: LoRA via Unsloth/PEFT aplicada em módulos projetores da atenção/MLP.
- Treinador: TRL SFTTrainer (instrução supervisionada) com formatação de prompts.
- Saída: dataset formatado `trn_output.json` e adaptadores/weights LoRA salvos ao final.

## Dados de treino (trn.json)
- Formato esperado (JSONL): cada linha contém um objeto com os campos:
  - `title`: título do produto (string)
  - `content`: descrição do produto (string)
- Linhas com `title` vazio ou `content` vazio são descartadas.
- O notebook lê e grava, por padrão, nestes caminhos (Colab + Drive):
  - `DATA_PATH = "/content/drive/MyDrive/tc3/trn.json"`
  - `OUTPUT_PATH_DATASET = "/content/drive/MyDrive/tc3/trn_output.json"`

Exemplo de linha válida:
```json
{"title": "Girls Ballet Tutu Neon Pink", "content": "Uma linda saia tutu rosa neon para ballet, leve e confortável..."}
```

## Dependências e ambiente
No Google Colab, o notebook instala as dependências assim:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers==0.0.27 "trl<0.9.0" peft accelerate bitsandbytes
pip install -q transformers datasets triton torch torchvision xformers
```
Observações:
- É necessário montar o Google Drive para acessar os arquivos (célula `drive.mount('/content/drive')`).
- Use GPU com suporte adequado (o notebook comenta L4). O script utiliza 4-bit (`load_in_4bit=True`).

## Carregamento do modelo pré-treinado
O notebook carrega o modelo/ tokenizer com Unsloth:
```python
from unsloth import FastLanguageModel
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    device_map = "auto",
)
```
Antes e depois do treino, o notebook realiza testes de geração usando um prompt de instrução:
```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{}
```

## Preparação do dataset formatado
O notebook inclui funções para ler `trn.json` (JSONL) e gerar `trn_output.json` com o seguinte mapeamento:
- `instruction`: string fixa "Describe the product [input]"
- `input_text`: recebe o `title`
- `response`: concatena "The '{title}' is {content}"

Trecho relevante:
```python
formatted_item = {
    "instruction": "Describe the product [input]",
    "input_text": title,
    "response": f"The '{title}' is {content}"
}
```
Depois, é aplicado um template de prompt por amostra para o SFT:
```python
product_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
```

## Configuração LoRA (via Unsloth)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

## Hiperparâmetros de treino (TrainingArguments)
Definidos conforme o notebook:
- `per_device_train_batch_size = 8`
- `gradient_accumulation_steps = 4`
- `warmup_ratio = 0.2`
- `learning_rate = 3e-5`
- `max_steps = 60`
- `fp16 = not is_bfloat16_supported()`
- `bf16 = is_bfloat16_supported()`
- `logging_steps = 1`
- `optim = "adamw_8bit"`
- `weight_decay = 0.01`
- `lr_scheduler_type = "linear"`
- `seed = 3407`
- `output_dir = "outputs"`
- `num_train_epochs = 4`
- `save_strategy = "steps"`
- `save_steps = 10`

O treino é executado via `SFTTrainer`. O notebook também registra perdas de treino/validação com um callback customizado e plota um gráfico (matplotlib) ao final.

## Execução (passo a passo resumido)
1. Abra `tc3_fine_tuning.ipynb` no Google Colab.
2. Monte o Google Drive: `drive.mount('/content/drive')`.
3. Instale as dependências (células indicadas no notebook).
4. Garanta que `trn.json` esteja em `/content/drive/MyDrive/tc3/trn.json` no formato esperado.
5. Execute as células na ordem: pré-teste de geração, formatação de dataset, configuração LoRA, treino, gráfico de perdas e pós-teste.
6. Os arquivos gerados incluem `trn_output.json` e o modelo/adaptadores salvos.

## Teste de inferência (pós-treino)
O notebook demonstra novas gerações para títulos como `'On Happiness, U.S. Edition'` e `'The book of revelation'`, usando o mesmo prompt de instrução acima e `TextStreamer` para exibir a resposta.

## Salvamento do modelo
Ao final, os artefatos LoRA e tokenizer são salvos no Drive:
```python
model.save_pretrained("/content/drive/MyDrive/tc3/lora_model")
tokenizer.save_pretrained("/content/drive/MyDrive/tc3/lora_model")
```

## Dicas e observações
- Use uma GPU com VRAM suficiente e mantenha `load_in_4bit=True` para economia de memória.
- Se ocorrer OOM, reduza `per_device_train_batch_size`, aumente `gradient_accumulation_steps` e/ou reduza `max_seq_length`.
- Verifique caminhos do Drive (ajuste `/content/drive/MyDrive/tc3/` conforme sua pasta) e permissões.

## Estrutura relevante do repositório
- `tc3_fine_tuning.ipynb`: notebook principal deste projeto.
- `data/trn.json`: exemplo/local alternativo de dados (se não usar Drive, adapte o caminho no notebook).
- `outputs/`: diretório de saídas do treino (no ambiente de execução).
- `unsloth_compiled_cache/`: cache interno do Unsloth.

## Licença
Consulte o arquivo `LICENSE` deste repositório e as licenças dos modelos e bibliotecas utilizadas (Unsloth, Transformers, Datasets, TRL, PEFT, Accelerate, BitsAndBytes, etc.).