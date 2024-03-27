# End-to-end-RAG-Implementation-using-Amazon-Bedrock


## How to run?

```bash
conda create -n bedrockdemo python=3.8 -y
```

```bash
conda activate bedrockdemo 
```

```bash
pip install -r requirements.txt
```


### Install aws cli from the following link:
```bash
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```


### Add credentials by running the following command
```bash
aws configure
```


### To run streamlit app

```bash
streamlit run bedrock_test.py
```


```bash
streamlit run rag_demo.py
```