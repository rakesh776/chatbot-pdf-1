import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

import uuid
import pinecone
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_index_name = "langchain"
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')
pinecone_index = pinecone.Index(index_name=pinecone_index_name)
index = pinecone.Index("langchain")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(raw_text, file_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    vectors = []

    for chunk in text_splitter.split_text(raw_text):
        r = openai.Embedding.create(input=[chunk], model="text-embedding-ada-002")['data'][0]['embedding']
        # st.write(chunk)
        vectors.append({'id': str(uuid.uuid4()), 'values': r, 'metadata': {'content': chunk, 'filename': file_name}})

    upsert_response = index.upsert(vectors=vectors)
  
    return vectors


def embed_question(question):
    try:
        user_question_embedding = openai.Embedding.create(input=[question], model="text-embedding-ada-002")['data'][0]['embedding']

        return user_question_embedding
    except Exception as e:
        st.error(f"Error embedding question: {str(e)}")
        return None

prompt_prefix = """
        Sources:
        {sources}

        Chat History:
        {chat_history}

        Question:
        {question}
        """

system_prompt = """You are a helpful AI assistant. Use the following pieces of context to answer the Question at the end.
      If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
      If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. Be brief in your answers.
      Answer ONLY with the facts listed in the list of Sources and Chat History below.
      For tabular information return it as an html table. Do not return markdown format.
      Each Source and Chat History has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [coca-cola-2015.pdf]. Don't combine sources, list each source separately, e.g. [coca-cola-2018.pdf][pepsi-2020.pdf].  
      You can also use Chat History provided after the sources, to answer question."""

overrides = {}

def handle_userinput(user_question):
    user_question_embedding = embed_question(user_question)

    if user_question_embedding is not None:
        results = index.query(vector = user_question_embedding, top_k=16, include_metadata=True)

        points = []
        for match in results.get("matches", []):
            metadata = match.get('metadata')
            points.append(metadata)
        # st.write(points)

        results = [doc['filename'] + ":: " + doc['content'] + "\n" for doc in points]
        content = "\n".join(results)

        prompt = prompt_prefix.format(injected_prompt="", sources=content, chat_history="", question=user_question)

        completion = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k-0613',
                                                  messages=[{"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": prompt}],
                                                  temperature=overrides.get("temperature") or 0, max_tokens=512, n=1, )

        st.write(completion.choices[0].message.content)
        return completion.choices[0].message.content

    else:
        st.error("Failed to embed the question.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDFs")
    user_question = st.text_input("Ask a question?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("your documents")
        pdf_docs = st.file_uploader(
            "upload PDF here n click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                for pdf_file in pdf_docs:
                    raw_text = get_pdf_text([pdf_file])

                    get_vectorstore(raw_text, pdf_file.name)
                


if __name__ == '__main__':
    main()
