#/bin/bash
#

apt-get update
apt-get install unzip jq git vim wget
pip install --upgrade pip
pip install numpy==1.26.4 scipy numba pandas
pip install GPUtil azureml azureml-core tokenizers ninja cerberus sympy sacremoses sacrebleu==1.5.1 sentencepiece scipy scikit-learn "urllib3<2"

git clone -b main http://github.com/rocm/transformers/ transformers
cd transformers/
pip install -e .
sed -i 's$git+https://github.com/huggingface/accelerate@main#egg=accelerate$$g' ./examples/pytorch/_tests_requirements.txt
cd examples/pytorch/
pip install -r _tests_requirements.txt
cd ../../../

pip install accelerate datasets diffusers huggingface_hub peft
pip install --upgrade botocore boto3
pip install --upgrade urllib3 six evaluate

git clone https://github.com/karpathy/minGPT.gi
cd minGPT/
pip install .
cd ..
