# !pip install langchain -q
# !pip install chromadb -q
# !pip install pypdf -q
# !pip install sentence-transformers -q
# !pip install google-generativeai -q
# !pip install langchain-community -q
# !pip install pyTelegramBotAPI -q

GEMINI_API_KEY = "AIzaSyC9_igugvIzqj3Zm2NDAENIFczjw7gCfKk"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loaders = [PyPDFLoader('Protei_SSW5.pdf')]

docs = []

for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

vectorstore = Chroma.from_documents(docs, embeddings_function, persist_directory="./chroma_db_nccn")

print(vectorstore._collection.count())

import os
import signal
import sys
import google.generativeai as genai


def signal_handler(sig, frame):
    print('\nT============')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

def generate_rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("''", "").replace('""', "").replace("\n", " ")
    prompt = ("""
    You are a helpful and informative bot that answers questions using text from the reference context included below. \
    Be sure to respond in a complete sentence being comprehensive including all relevant background information. \
    The context contains information from PROTEI SSW 5 multiservice access switches manual. Write russian only. \
    If you have the answear write "Не спам", and after that on the next line write your full answear. \
    If you don't know the answear, just say "Спам: Вопрос не по теме". Don't try to make up an answer. \
    If the context is irrelevant to the answer, just say "вопрос не по теме".
      QUESTION: '{query}'
      =========
      CONTEXT: '{context}'
      =========
      Answer:
              """).format(query=query, context=context)
    return prompt


def get_relevant_context_from_db(query):
    context = ""
    embeddings_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_db = Chroma(persist_directory="/content/chroma_db_nccn", embedding_function=embeddings_function)
    search_results = vector_db.similarity_search(query, k=8)
    for result in search_results:
        context += result.page_content + "\n"
    return context


def generate_answer(prompt):
    genai.configure(api_key = GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    answer = model.generate_content(prompt)
    return answer.text

import telebot
from telebot import types

while True:
    print('--------------------------------------------------------------------------------------------------------------\n')
    print("Задайте запрос?\nДля завершения выполнения программы впишите "'esc'"." )
    query = input("Запрос: ")
    context = get_relevant_context_from_db(query)
    if query.lower() == "esc":
        print('\nСпасибо за пользование моей программой.')
        break
    prompt = generate_rag_prompt(query=query, context=context)
    answer = generate_answer(prompt)
    print(prompt)
    print('--------------------------------------------------------------------------------------------------------------\n')
    print(answer)

# Интеграция с Telegram ботом
bot = telebot.TeleBot('') #токен телеграмм бота


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup()
    btn2 = types.KeyboardButton('User id')
    markup.row(btn2)
    bot.send_message(message.chat.id, 'Здравствуйте, я - бот, который содержит руководство по настройке мультисервисного коммутатор доступа PROTEI SSW 5.\
Он анализирует поступающих вопрос от пользователя, пытается найти соответствие с руководством и определеяет, является ли сообщение спамом. Ссылка на документацию: https://protei.ru/documentation', reply_markup=markup)



@bot.message_handler()
def info(message):
    if message.text.lower() == 'id' or message.text.lower() == 'user id':
        bot.reply_to(message, f'ID: {message.from_user.id} пользователя '
                              f'{message.from_user.last_name} {message.from_user.first_name}')

    else:
        query = message.text
        context = get_relevant_context_from_db(query)
        prompt = generate_rag_prompt(query=query, context=context)
        answer = generate_answer(prompt)
        bot.send_message(message.chat.id, answer)
bot.polling(none_stop=True)