from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage


class MemoryCallbackHandler(BaseCallbackHandler):
    def __init__(self, memory):
        self.message = ""
        self.memory = memory
        self.user_input = ""  # 사용자 입력을 저장할 변수 추가

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Chain이 시작될 때 사용자 입력 저장
        if "input" in inputs:
            self.user_input = inputs["input"]

    def on_llm_start(self, *args, **kwargs):
        self.message = ""

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token


# 메모리 생성
memory = ConversationBufferMemory()

# MemoryCallbackHandler 인스턴스 생성
memory_callback_handler = MemoryCallbackHandler(memory)


# 메모리에 저장된 대화 내용 확인
print(memory.load_memory_variables({}))


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def main():

    vectorstore = FAISS.load_local(
        "vector_stores/randomteam_blog",
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
            memory_callback_handler,
        ],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                Context: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    send_message("안녕하세요!", "ai", save=False)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    paint_history()

    message = st.chat_input("질문을 입력해주세요.")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    if message:
        send_message(message, "human")
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

    else:
        st.session_state["messages"] = []


def main_ui():

    st.set_page_config(
        page_title="랜덤팀 블로그 스크래핑",
        page_icon="🎉",
    )

    st.title("랜덤팀 블로그 스크래핑")


if __name__ == "__main__":
    load_dotenv()
    main_ui()
    main()
