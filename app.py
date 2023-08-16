import streamlit as st
from streamlit_chat import message
import tempfile   # temporary file
from langchain.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain.embeddings import HuggingFaceEmbeddings # import hf embedding
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss' # # Set the path of our generated embeddings


# Loading the model of your choice
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    # the model defined, can be replaced with any ... vicuna,alpaca etc
    # name of model
    # tokens
    # the creativity parameter
    return llm


st.title("Llama2 Chat CSV - 🦜🦙")
#st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href=''></a></h3>",unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload File", type="csv") # uploaded file is stored here
# file uploader
if uploaded_file:
    # tempfile needed as CSVLoader accepts file_path exclusively
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name # save file locally

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','}) # any loader can be put here based on the data being used
    # csv_args={'delimiter': ','} for faulty formatted csv
    data = loader.load() # load the data
    #st.json(data)   # uncomment to check uploaded data
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'}) # use sentence transformer to create embeddings

    # FAISS Can be replaced by Chroma... so it will be like CHROMA.fromdocuments...
    db = FAISS.from_documents(data, embeddings) # pass data embeddings vector data here
    db.save_local(DB_FAISS_PATH) # save vector embedding here on mentioned path
    llm = load_llm() # Load the Language model here

    # the conversational chain which preserves context learning in chat
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    # ConversationalRetrievalChain can be replaced by LLMChain,retrivalQA

    # func for streamlit chat takes query from User
    def conversational_chat(query):
        # key value pairs of conversational history
        result = chain({"question": query, "chat_history": st.session_state['history']}) # enduser query and result variable

        # add all responses here with query to preserve context
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"] # get the generated result


    # appending history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Start message, in context of no question having being not asked yet
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # container for the chat history
    response_container = st.container() # form

    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input') # user input values are here
            submit_button = st.form_submit_button(label='Send') # button to retrieve answer

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input) # old user input is appended
            st.session_state['generated'].append(output) # append the generated

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")





