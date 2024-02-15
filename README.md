# Sentiment Analysis Flask-App
>This repo consists of sentiment detection code based client server architecture. Flask is used for server implementation. Currently, there are two LLMs implemented on the server:
- DistilBERT: [hugging face link](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
- XLM-RoBERTa: [hugging face link](https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta)

>Run server
```
python -m flask run 
```

## Screenshot
<img width="1475" alt="Sentiment Detection Flask-App" src="https://github.com/sofiahalima/sentiment-analysis-bert/assets/26790739/b8289c92-1919-42cf-a3a9-e3e8a3b6df7c">

> [!TIP]
> To add more models, browse pretrained LLMs from: [hugging face text-classification](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending).
