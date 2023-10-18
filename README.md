# Chat-GPT-Custom-LLM-Demo
Chat GPT Custom LLM Demo

This is a simple demo of Chat GPT Custom domain LLM and prompt demo. 


## Install the below libraries

python3 -m pip install -U pip   
pip3 install openai                                       
pip3 install PyPDF2
pip3 install langchain
pip3 install llama-index
pip3 install gradio
pip3 install transformers    
pip3 install docx2txt       
pip3 install protobuf   
pip3 install sentencepiece             
pip3 install torch             
pip3 install gpt_index
pip3 install PyCryptodome
pip3 install langchain langchain-experimental       
pip3 install torch transformers python-pptx Pillow

## Add files for Indexing

Copy txt, pdf, excel, docx files into docs folder, if you use a existing folder, make sure you dont have CSV files or other files, removed, since all files are not indexable by this code.

## RUN python3 model.py 

## Access via Web Browser

If you dont get any error (you are lucky !!!), a localhost link is created, open this link and ask questions to Chat GPT on the files you have put in docs folder.

Running on local URL:  http://127.0.0.1:7860

### Reference URL

Tutorial -> The link has old functions which has changed, check the second link for new functions list, MK has fixed the code
as on Oct 17, 2023, but the code seems to change fast, some within 1 month, so might not work in future.

https://medium.com/rahasak/creating-custom-chatgpt-with-your-own-dataset-using-openai-gpt-3-5-model-llamaindex-and-langchain-5d5837bf9d56

Reference documentation to fix the Code in case of errors.
https://gpt-index.readthedocs.io/en/stable/end_to_end_tutorials/usage_pattern.html
