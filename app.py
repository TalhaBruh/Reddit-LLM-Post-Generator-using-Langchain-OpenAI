import json
import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from utils.helper import progress_bar_map
from utils.custom import css_code

load_dotenv(find_dotenv())
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def search_for_articles(query: str) -> dict:
    """
    This function is used to search for articles using the serpapi
    :param query: query used to search
    :return: search results
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        "X-API-KEY": SERPAPI_API_KEY,
        "Content-Type": "application/json"
    }
    response: Any = requests.request("POST", url, headers=headers, data=payload)
    response_data: dict = response.json()

    return response_data


def find_the_best_article_urls(response_data: dict, query: str) -> list:
    """
    This function is used to find the best url using a prompt template and the gpt-3.5-turbo model
    :param response_data: dictionary of search results
    :param query: query used to search
    :return: list of the top url
    """
    response_str = json.dumps(response_data)
    prompt_template: str = """
    You're a world class researcher, and are very good at finding the most relevant articles for certain topics;
    {response_str}
    Above is a list of search results for the query {query}.
    Please choose the best article from the list, return ONLY an array of the url, do not include anything else;
    """
    llm: Any = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["response_str", "query"])

    article_finder: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)

    generated_urls = article_finder.predict(response_str=response_str, query=query)

    url_list = json.loads(generated_urls)
    print(url_list)

    return url_list


def get_content_from_urls(urls: list) -> list:
    """
    This is a function that is used to fetch the data from the urls, and pass it
    to the LLM for summary
    :param urls: list of urls
    :return: list of retrieved data
    """
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data


def summarise_content(data: str, query: str) -> list[str]:
    """
    This function is used to summarise the fetched data, using the gpt-3.5-turbo and prompt template
    :param data: data to summarise
    :param query: query used to search
    :return: list of summarised text
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(data)

    llm: Any = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    prompt_template: str = """
    {text}
    You're a world class researcher, and you'll try to summarise the text above in order to create a reddit post {query}
    Please follow all of the following rules when summarising:
    1/ Make sure the content is engaging information with good data
    2/ Make sure the content is not too long, it should be no more than 40,000 characters
    3/ The content should address the {query} topic very well
    4/ The content needs to be viral, and get at least 1000 likes
    5/ The content needs to written in a way that is easy to read and understand
    6/ The content needs to give the reader some actionable advice and insights
    
    SUMMARY:
    """

    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["text", "query"])

    llm_summariser: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)

    summaries: list = []

    for chunk in enumerate(text):
        print(chunk)
        summary = llm_summariser.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries


def generate_reddit_post(summaries: list, query: str) -> str:
    """
    This function is used to feed the summaries into an LLM, to generate a Reddit post using a prompt template
    :param summaries: summarised text
    :param query: query used to search
    :return: generated text to post
    """
    summaries_str = str(summaries)

    llm = Any = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    prompt_template = """
    {summaries_str}
    You are a world class researcher and reddit user with a large amount of karma, The text above is some content about
    {query}. Please write a reddit post about {query} using the text above, and following all the rules below:
    1/ The post needs to be engaging, informative with good data
    2/ Make sure the post is not too long, it should be no more than 40,000 characters
    3/ The post should address the {query} topic very well
    4/ The post needs to be viral, and get at least 1000 likes
    5/ The post needs to written in a way that is easy to read and understand
    6/ The post needs to give the reader some actionable advice and insights
    
    REDDIT POST:
    """

    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["summaries_str", "query"])

    reddit_post_chain: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)

    reddit_post: Any = reddit_post_chain.predict(summaries_str=summaries_str, query=query)

    return reddit_post


def progress_bar(user_feedback: list[str], amount_of_time: int = 20) -> None:
    """
    This function is used to create a progress bar to improve the UX
    :param user_feedback: messages to output with the progress bar
    :param amount_of_time: time taken to sleep
    :return: None
    """
    with st.status("AI models hard at work", expanded=True) as status:
        st.write(user_feedback[1])
        time.sleep(amount_of_time)
        st.write(user_feedback[2])
        time.sleep(1)
        status.update(label=f"Download {user_feedback[0]} of 5 complete!", state="complete", expanded=False)


def main() -> None:
    """
    Main function
    :return: None
    """
    st.set_page_config(page_title="Reddit post generator", page_icon="img/webworks87-favicon-light.png", layout="wide")

    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.image("img/webworks87-light-logo.jpg")
        st.write("---")
        st.write("App created by James Aymer")

    st.header("Generate a Reddit post")
    query = st.text_input("Enter a topic for the Reddit post")

    if query:
        search_results = search_for_articles(query)
        progress_bar(user_feedback=progress_bar_map.get("search_for_articles"))

        urls = find_the_best_article_urls(search_results, query)
        progress_bar(user_feedback=progress_bar_map.get("find_the_best_article_urls"))

        data = get_content_from_urls(urls)
        progress_bar(user_feedback=progress_bar_map.get("get_content_from_urls"))

        summaries = summarise_content(data, query)
        progress_bar(user_feedback=progress_bar_map.get("summarise_content"))

        post = generate_reddit_post(summaries, query)
        progress_bar(user_feedback=progress_bar_map.get("generate_reddit_post"))

        with st.expander("Search results"):
            st.info(search_results)
        with st.expander("Best url"):
            st.info(urls)
        with st.expander("Data"):
            st.info(data)
        with st.expander("Summaries"):
            st.info(summaries)
        with st.expander("Post"):
            st.info(post)


if __name__ == "__main__":
    main()
