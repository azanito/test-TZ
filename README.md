# Дообучение Mistral-7B через LoRA

## Описание

Дообучал языковую модель Mistral-7B на датасете инструкций, чтобы она лучше следовала формату вопрос-ответ. Использовал QLoRA — это позволяет обучать модель на GPU с небольшим объёмом памяти (хватает T4 в Google Colab).

## Что сделано

- подготовлен датасет ~500 примеров из tatsu-lab/alpaca
- дообучена модель через LoRA (обучается ~0.1% параметров)
- проведена оценка — сравниваются ответы базовой и дообученной модели

## Модель

Mistral-7B-v0.1, загружается в 4-bit (QLoRA через bitsandbytes)

LoRA параметры:
- r = 16
- alpha = 32
- target_modules: q_proj, v_proj

## Метрика

ROUGE-L — считается между ответами базовой и дообученной модели как сигнал расхождения.

Средний результат: ~0.36

## Вывод

После дообучения модель стала лучше следовать формату `### Instruction / ### Response`, ответы стали более структурированными и по делу. Базовая модель часто продолжает промпт вместо того чтобы отвечать — дообученная этого не делает.

Улучшение небольшое, потому что датасет маленький (500 примеров) и всего 1-2 эпохи. На большем датасете результат был бы лучше.

## Как запустить

### Google Colab (рекомендуется)

1. Включить GPU: Runtime → Change runtime type → T4 GPU
2. Запустить:

```python
!git clone https://github.com/your-username/qlora-finetuning
%cd qlora-finetuning
!pip install -r requirements.txt
!python prepare_data.py
!python train.py --num_train_epochs 1
!python evaluate.py
```

Или одной командой:
```python
!bash run_colab.sh
```

### Локально

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate.py
```

## Структура проекта

```
├── data/dataset.jsonl          # датасет
├── output/lora-adapter/        # сохранённый адаптер
├── evaluation/results.json     # результаты оценки
├── prepare_data.py
├── train.py
├── evaluate.py
├── requirements.txt
└── run_colab.sh
```
