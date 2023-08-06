import streamlit as st
from PIL import Image
from dotenv import load_dotenv # for accessing the api keys in the env file
from PyPDF2 import PdfReader # for reading pdfs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,  LlamaCppEmbeddings # embed words
from langchain.vectorstores import FAISS # for storing vectors
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
# put all pdfs into one paragraph
def read_all_texts(pdfs):
    temp = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            temp += page.extract_text()
    return temp

# divide the huge paragraph into chunks
def chunk(txt):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(txt)
    return chunks

def store_vectors(chunks):
    # embeddings = OpenAIEmbeddings()
    # or using Meta's llm:  Llama2
    embeddings = LlamaCppEmbeddings(model_path="llama-2-7b-chat.ggmlv3.q2_K.bin") 
    storage = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return storage

def conversation_chain(vector_store):
    #llm  = ChatOpenAI() # ChatGPT or Llama2
    callback_manager =  CallbackManager([StreamingStdOutCallbackHandler()]) # for Llama2
    llm = LlamaCpp(model_path="llama-2-7b-chat.ggmlv3.q2_K.bin",callback_manager=callback_manager,verbose=False,n_ctx=2048)
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    ) # chat with our contexts/vectors and have memory
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    # st.write(response) for testing
    st.session_state.chat_history = response['chat_history']
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(msg.content)
                
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
            
def main():
    load_dotenv()
    st.set_page_config(page_title= "Your Chatbot", page_icon= ':robot_face:')
    # make sure streamlit will not reload
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    image = Image.open('/Users/William/Downloads/Data-Projects/customized_chatbot/image.png')
    st.image(image)
    st.title("Chat with Your Customized Chatbot!")
    st.subheader("What documents do you want to interact with?")
    pdfs = st.file_uploader('Upload your documents/PDFs here:', accept_multiple_files=True)
    button = st.button("Upload")
    user_question = st.text_input("Please ask a question about your documents AFTER the processing icon disappears.")
    
    if button:
        
        with st.spinner('Processing...'):
            # get pdf text
            raw_texts = read_all_texts(pdfs)
            # print(raw_texts)
            # make texts into chunks
            text_chunk = chunk(raw_texts)
            print(text_chunk)
            # create vector stores
            vector_storage = store_vectors(text_chunk)
            # create conversation chain for chat memory
            st.session_state.conversation = conversation_chain(vector_storage)
    
    if user_question:
        with st.spinner('Chatbot is answering...'):
            handle_userinput(user_question)            

if __name__ == "__main__":
    main()