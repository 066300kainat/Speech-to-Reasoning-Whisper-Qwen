# Speech-to-Reasoning Pipeline with Whisper & Quantized Qwen 2.5

An end-to-end AI pipeline that converts spoken questions into text using OpenAI Whisper and generates step-by-step logical answers using a 4-bit quantized Qwen 2.5 3B Instruct model with Unsloth.

## Project Overview

This project demonstrates how Automatic Speech Recognition (ASR) can be integrated with a Large Language Model (LLM) to build a speech-to-reasoning system.

The pipeline takes audio input, transcribes the speech into text, prepares the transcript as an LLM prompt, and generates a logical response.

The implementation uses Google Colab with a Tesla T4 GPU and 4-bit quantization to make LLM inference more memory-efficient.

## Key Features

- **Speech Recognition:** Converts spoken questions into text using OpenAI Whisper.
- **Audio Segmentation:** Displays transcription segments produced by Whisper.
- **Text Processing:** Applies Unicode normalization and UTF-8 encoding.
- **Quantized LLM:** Uses Qwen 2.5 3B Instruct in 4-bit quantized format.
- **Logical Reasoning:** Generates step-by-step answers to spoken questions.
- **Batch Processing:** Processes multiple text prompts in a single inference batch.
- **GPU Memory Monitoring:** Tracks GPU memory usage during model inference.
- **End-to-End Pipeline:** Connects speech recognition and LLM reasoning in one workflow.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Google Colab | Development environment |
| OpenAI Whisper | Speech-to-text transcription |
| Qwen 2.5 3B Instruct | Logical reasoning |
| Unsloth | Efficient quantized model loading and inference |
| PyTorch | Model inference and GPU operations |
| Transformers | Tokenization and text generation |
| BitsAndBytes 4-bit | Model quantization |
| FFmpeg | Audio processing |
| CUDA | GPU acceleration |

## Model Configuration

**Speech Recognition Model**
- Model: Whisper Base
- Task: Automatic Speech Recognition (ASR)

**Reasoning Model**
- Model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Quantization: 4-bit
- Maximum sequence length: 2048 tokens
- Inference mode: Enabled
- Generation: Greedy decoding

## System Architecture

```text
        Audio Input
             |
             v
      OpenAI Whisper
             |
             v
    Speech Transcription
             |
             v
   Text Normalization
     UTF-8 Encoding
             |
             v
      Prompt Creation
             |
             v
    Tokenization / Batch
             |
             v
   Qwen 2.5 3B Instruct
       4-bit Quantized
             |
             v
     Logical Reasoning
             |
             v
       Final Answer
```

## Implementation Workflow

### 1. Environment Setup
- Configured Google Colab with a Tesla T4 GPU.
- Installed Whisper, Unsloth, and required dependencies.
- Verified CUDA availability and GPU memory.

### 2. Audio Transcription
- Loaded the Whisper Base model.
- Provided audio files as input.
- Generated text transcriptions.
- Retrieved individual transcription segments.

### 3. Text Processing
- Normalized transcription text using Unicode normalization.
- Encoded and decoded text using UTF-8.
- Prepared structured prompts for the reasoning model.

### 4. Quantized LLM Loading
- Loaded Qwen 2.5 3B Instruct using Unsloth.
- Enabled 4-bit quantization.
- Configured the model for inference.
- Verified quantized Linear4bit modules.

### 5. Reasoning and Batch Inference
- Passed Whisper transcriptions to Qwen.
- Generated step-by-step logical answers.
- Processed multiple prompts together using batch tokenization and inference.

### 6. GPU Memory Monitoring
- Monitored GPU memory allocation.
- Evaluated memory usage during inference.

## Sample Results

The pipeline was tested using two separate audio questions.

| Audio Input | Question | Expected Answer | Model Result |
|---|---|---|---|
| Audio 1 | A person has 5 apples and gives away 2. | 3 apples | 3 apples |
| Audio 2 | A shopkeeper has 24 oranges and sells 9 in the morning and 7 in the evening. | 8 oranges | 8 oranges |

### Sample Model Output

**Audio 1 – Apple Problem**

```text
Initial apples = 5
Apples given away = 2

Apples left = 5 - 2

Final Answer: 3 apples
```

**Audio 2 – Orange Problem**

```text
Initial oranges = 24
Morning sales = 9
Evening sales = 7

Remaining oranges = 24 - 9 - 7

Final Answer: 8 oranges
```

## Performance Results

| Metric | Observed Result |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 14.56 GB total |
| Quantized Model | Qwen 2.5 3B Instruct |
| Quantization | 4-bit |
| Linear4bit Modules | 252 |
| GPU Allocated Memory | Approximately 1.99 GB |
| Audio Files Tested | 2 |
| Batch Reasoning Tests | Passed |

*GPU memory allocation is the observed value from the completed batch inference test and may vary between runs.*

## Important Note About Transcription

During testing, Whisper incorrectly recognized the word "nine" as "iron" in the second audio transcription. The transcript was manually corrected using the original generated audio text before passing it to the reasoning model.

This correction is documented to distinguish raw ASR output from the corrected input used for reasoning.

## How to Run

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Choose **T4 GPU**.
4. Run the installation and setup cells.
5. Upload your audio files.
6. Run the Whisper transcription cells.
7. Load the quantized Qwen model.
8. Execute the end-to-end speech-to-reasoning pipeline.
9. Run the batch inference and GPU memory verification cells.

## Hardware Requirements

- Google Colab or a compatible Python environment.
- NVIDIA GPU recommended.
- CUDA-compatible PyTorch environment.
- Internet connection for downloading pretrained models.

## Project Outcome

Successfully implemented an end-to-end speech-to-reasoning pipeline integrating Whisper ASR with a 4-bit quantized Qwen 2.5 3B Instruct model.

The project demonstrates audio transcription, text processing, quantized LLM inference, batch processing, logical reasoning, and GPU memory monitoring in a single workflow.

## Author

**Kainat Zafar**

- GitHub: [066300kainat](https://github.com/066300kainat)
- LinkedIn: [Kainat Zafar](https://www.linkedin.com/in/kainat-zafar-17151a35a/)

---

*Built as an AI/LLM engineering implementation project using Whisper, Unsloth, and Qwen 2.5.*
