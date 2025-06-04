# Llama 3.2 1B QLoRa para Regresión de Péptidos

Este proyecto ajusta el modelo Llama 3.2 1B usando QLoRa (Quantized Low-Rank Adaptation) para tareas de regresión que predicen valores de actividad de péptidos basados en secuencias de aminoácidos.

## Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Instalación](#instalación)
3. [Preparación de Datos](#preparación-de-datos)
4. [Notación de Secuencias de Péptidos](#notación-de-secuencias-de-péptidos)
5. [Configuración del Entrenamiento](#configuración-del-entrenamiento)
6. [Resultados](#resultados)

## Requisitos Previos

1. **Cuenta de Hugging Face**:

   - Cree una cuenta en [huggingface.co](https://huggingface.co)
   - Genere un token de escritura en la configuración de la cuenta
   - Use `notebook_login()` para autenticación

2. **Acceso al Modelo**:

   - Solicite acceso a `meta-llama/Llama-3.2-1B` en Hugging Face Hub

3. **Google Drive**:

   - Se utiliza para almacenamiento de datos y persistencia del modelo

4. **Requisitos de Hardware**:
   - GPU con suficiente VRAM (probado en Google Colab con GPU T4)

## Instalación

```bash
pip install -q --upgrade transformers peft trl datasets accelerate bitsandbytes scikit-learn pandas
```

Con versiones específicas en archivo `requirements.txt`:

## Preparación de Datos

Los datos deben estar en formato CSV con columnas:

- `sequence`: Secuencias de aminoácidos de péptidos (en notación completa)
- `label`: Valor de actividad (objetivo de regresión)

Ejemplo de estructura del dataset:

```csv
sequence,label
Alanine-Alanine-Valine-Alanine,0.75
Leucine-Methionine-Asparagine-Proline,1.24
...
```

## Notación de Secuencias de Péptidos

Las secuencias de péptidos utilizan la notación completa de nombres de aminoácidos separados por guiones. Esta es la correspondencia de códigos de una letra a nombres completos:

| Código | Aminoácido    |
| ------ | ------------- |
| A      | Alanine       |
| R      | Arginine      |
| N      | Asparagine    |
| D      | Aspartic acid |
| C      | Cysteine      |
| Q      | Glutamine     |
| E      | Glutamic acid |
| G      | Glycine       |
| H      | Histidine     |
| I      | Isoleucine    |
| L      | Leucine       |
| K      | Lysine        |
| M      | Methionine    |
| F      | Phenylalanine |
| P      | Proline       |
| S      | Serine        |
| T      | Threonine     |
| W      | Tryptophan    |
| Y      | Tyrosine      |
| V      | Valine        |

Ejemplo: La secuencia "AARG" se representa como "Alanine-Alanine-Arginine-Glycine"

## Configuración del Entrenamiento

### Modelo Base

```python
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
)
```

### Configuración QLoRa

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Parámetros de Entrenamiento

```python
training_args = TrainingArguments(
    output_dir="./llama3-regression",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=50,
    fp16=True
)
```

## Resultados

Tras el entrenamiento, el modelo se evaluó en un conjunto de prueba con los siguientes resultados:

- RMSE: 0.46
- MAE: 0.37
- R²: 0.48
