# UNICORN Internal Development 🦄💻

This repository contains the baseline code for various tasks in the UNICORN challenge.

## Repository Structure 🗂️

The repository is structured as follows:
```
├── vision                    # Baseline code for vision tasks
│   ├── encoding              # Code for image encoding
│   └── decoding              # Code for few-shot fine-tuning and adapters
├── language                  # Code for language tasks
├── vision-language           # Code for vision-language tasks
├── evaluation                # Code for task evaluation and metric computation
├── example-data              # Examples of interfaces and sample files
├── unicorn_io.py             # I/O code shared across tasks
├── encoding.py               # Script to go from inputs to either embeddings (vision) or predictions (language, vision-language)
└── evaluate.py               # Script to go from embeddings (vision) or predictions (language, vision-language) to metric
```

Want to implement a new task? Use the vision/language/vision-language folders to dump your utility code.
Then update the `process_input` function from `unicorn_io.py`. 
The following files should mostly stay unchanged: `encoding.py` & `evaluate.py`.

## Notes ⚠️

The code in this repository is under active development. New features and tasks are being added regularly.
Check the ``example-data`` folder for reference files to guide your development and ensure proper input/output formats.
