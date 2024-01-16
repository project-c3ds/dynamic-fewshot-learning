# dynamic-fewshot-learning

This repository contains the code for the paper [Dynamic Few-shot Learning for Computational Social Science]. 

## Code Structure
The code for this project is organized as follows:
- `benchmarks/` contains the code for the benchmark datasets used in the paper. This folder also contains the code for the data preprocessing and the code for the LLM analysis.
- `isca/` contains the code for the tools and utils needed to run the experiments.
- `benchmarks/data/` contains the data for the benchmark datasets used in the paper.
- `benchmarks/prep.ipynb` contains the code for the data preprocessing. (not needed to run the experiments)

### Experiment run

To run the experiments, you can use the following command in the terminal:

```bash
cd benchmarks
python <benchmark_name>.py --model_name <model_name> --temperature <temperature> --max_tokens <max_tokens>
```

where `<benchmark_name>` is the name of the benchmark dataset, `<model_name>` is the name of the model, `<temperature>` is the temperature value (default is 0), and `<max_tokens>` is the maximum number of tokens (default is 100).