# Text Normalization

A machine learning project for converting written text expressions into appropriate spoken forms.

## Project Overview

This project tackles the challenge of text normalization for speech and language applications. Text normalization is the process of converting written expressions like "12:47" and "$3.16" into their spoken forms ("twelve forty-seven" and "three dollars, sixteen cents" respectively). This is a critical component for:

- Text-to-speech synthesis (TTS)
- Automatic speech recognition (ASR)
- Other natural language processing applications

Instead of manually developing complex grammar rules for each language, this project uses machine learning algorithms to automate the text normalization process.

## Problem Statement

Given a corpus of text where each token has a "before" (raw text) and "after" (normalized text) form, our task is to predict the normalized form of text tokens in the test set.

## Dataset Description

The dataset consists of:

- **sentence_id**: Identifier for each sentence
- **token_id**: Identifier for each token within a sentence
- **before**: Raw text (input)
- **after**: Normalized text (target output)
- **class**: Token type category (available only in training data)
- **id**: Concatenation of sentence_id and token_id (e.g., "123_5")

## Getting Started

### Prerequisites

- Python 3.12
- Required libraries: pandas, pytorch, sklearn, numpy

### Installation

```bash
# Clone the repository
git clone https://github.com/HuskyDevClub/TextNormalization.git
cd TextNormalization
```

### Data Preparation

1. Download the dataset files:
   - en_train.csv: Training data with normalized text
   - en_test.csv: Test data without normalized text
   - en_sample_submission.csv: Submission format example

2. Place these files in the `./` directory.

## Results

Our best model achieves ~93% on the test set.

## Future Work

- Improve handling of rare token types
- Experiment with larger pre-trained language models
- Extend the approach to other languages
- Create a web-based demo

## Contributors

- Wynter Lin
- Danny Yue
- Jiani Ji
- Lyndsie Phan

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

⚠️ **ACADEMIC INTEGRITY WARNING** ⚠️

This project was created as a class assignment. If you use any part of this code or the results generated from it:

1. You **MUST** provide clear and proper credit to **all contributors** in:
   - Class presentations
   - Written essays or reports
   - Any derivative work
2. Failure to provide appropriate attribution may constitute **academic dishonesty** and/or **plagiarism**, which could result in academic penalties according to your institution's policies.
3. While you may reference and learn from this work, direct copying without attribution is **strictly prohibited**.

## References

Here is some code we examined and incorporated while developing my solution:

- Text tokenized and General Code Structure: https://github.com/bentrevett/pytorch-seq2seq 
- The Seq2Seq-Encoder-Decoder Model: https://github.com/312shan/Text-Normalization-in-pyTorch
