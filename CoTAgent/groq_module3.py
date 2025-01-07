# %%
import os
import json
import requests
from groq import Groq
import datetime
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

file_path = '../../API_keys/keys.json'
with open(file_path, 'r') as file:
    api = json.load(file)

def news_search(query, days_back=3):
    """
    Fetch news using NewsAPI and scrape full article content.
    Input: query - The search query for fetching news.
           days_back - Number of days back to fetch news (up to 3 days ago).
    Output: A string containing the formatted full news articles.
    """
    known_sources = ["bbc", "cnn", "geeky-gadgets", "reuters", "forbes"]
    base_url = "https://newsapi.org/v2/everything"
    results = ""

    for day in range(days_back, 0, -1):
        from_date = (datetime.datetime.now() - datetime.timedelta(days=day)).strftime('%Y-%m-%d')
        to_date = (datetime.datetime.now() - datetime.timedelta(days=day-1)).strftime('%Y-%m-%d')
        for page in range(1, 3):  # Fetch 2 pages for each day
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'sortBy': 'publishedAt',
                'apiKey': api['news_api'],
                'language': 'en',
                'pageSize': 20,
                'page': page
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                news_data = response.json()
                articles = news_data['articles']
                for article in articles:
                    source_name = article['source']['name'].lower()
                    url = article['url'].lower()
                    if any(known_source in source_name or known_source in url for known_source in known_sources):
                        if article['description'] and article['description'] != '[removed]':
                            title = article['title']
                            published_date = article['publishedAt'][0:10]
                            url = article['url']

                            # Fetch the full article content using BeautifulSoup
                            full_content = fetch_full_article_content(url)

                            results += (f"\nPublished date: {published_date}\n"
                                        f"Title: {title}\n"
                                        f"Full content: {full_content}\n"
                                        "--------------------\n")
            else:
                results += f"Failed to fetch news articles. Status code: {response.status_code}\n"
    return results

def fetch_full_article_content(url):
    """
    Fetch the full content of an article from the provided URL.
    Input: url - The URL of the article.
    Output: The full content of the article as a string.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Set up retry strategy
    retry_strategy = Retry(
        total=3,  # Retry up to 3 times
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    try:
        article_response = http.get(url, headers=headers, timeout=10)
        article_response.raise_for_status()
        soup = BeautifulSoup(article_response.content, 'html.parser')

        # Extract only the main content and avoid irrelevant texts
        paragraphs = soup.find_all('p')
        full_content = ' '.join([para.get_text(strip=True) for para in paragraphs if len(para.get_text(strip=True)) > 50])

    except requests.exceptions.RequestException as e:
        full_content = f"Could not retrieve full content. Error: {e}"
    except Exception as e:
        full_content = f"Error fetching full content: {e}"

    return full_content

def serp_search(query):
    """
    Search the web using SerpAPI.
    Input: question - The query string you want to search for.
    Output: result - A string containing relevant search information.
    """
    # Set up the search endpoint and parameters
    api_key = api['serp_api']
    search_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "num": 3  # Limit to 3 results for brevity
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        search_results = response.json()
        # Format the result summary
        results = search_results.get('organic_results', [])
        result_summary = "\n".join([f"Title: {res.get('title')}, Link: {res.get('link')}" for res in results])
        if not result_summary:
            result_summary = "No relevant results found."
        return result_summary
    except Exception as e:
        return f"An error occurred while searching SerpAPI: {str(e)}"
    
def google_search(query):
    """
    Search the web using Google Custom Search API.
    Input: question - The query string you want to search for.
    Output: result - A string containing relevant search information.
    """
    # Set up the search endpoint and parameters
    api_key = api['google_search']  # Replace 'google' with the actual key for Google Custom Search API
    cse_id = api['google_cse']  # Custom Search Engine ID
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": 10  # Limit to 3 results for brevity
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        search_results = response.json()
        # Format the result summary
        results = search_results.get('items', [])
        result_summary = "\n".join([f"Title: {res.get('title')}, Link: {res.get('link')}" for res in results])
        if not result_summary:
            result_summary = "No relevant results found."
        return result_summary
    except Exception as e:
        return f"An error occurred while searching Google API: {str(e)}"

def groq_llama3(system_role, user_message, model="llama-3.3-70b-versatile", functioncall = False, stop_s=None, verbose=False):
    """
    input : system role = string(explain about system role)
            user_message = string(default user message)
    output : string
    """
    import logging
    import os

    os.environ["HTTPX_LOG_LEVEL"] = "WARNING"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx").disabled = True
    logging.basicConfig(level=logging.WARNING)

    os.environ["GROQ_API_KEY"] = api['groq']
    client = Groq()

    MODEL = model
    messages = [
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": system_role
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": user_message,
        }
    ]
    if functioncall :
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            temperature=0.5,
            max_tokens=4096,
            tools = tools,
            tool_choice = 'auto',
            top_p=1,
            stop=stop_s,
            stream=False,
        )
        response_message = chat_completion.choices[0].message
        tool_calls = response_message.tool_calls
        if functioncall and tool_calls:
            available_functions = {
                "google_search": google_search, 
                "serp_search" : serp_search,
                "news_search" : news_search,
            }
            messages.append(response_message)

            # Handle only the first tool call to prevent iteration
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            try:
                function_to_call = available_functions[function_name]
            except KeyError:
                return f"Function `{function_name}` not available."
            
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for function arguments."
            
            try:
                function_response = function_to_call(**function_args)
            except Exception as e:
                function_response = f"An error occurred while executing `{function_name}`: {str(e)}"
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

            if verbose == True:
                if function_name == 'news_search':
                    print(f"Function called: {function_name}")
                    print(f"Function arguments: {function_args}")
                    print('-'*20)
                else:
                    print(f"Function called: {function_name}")
                    print(f"Function arguments: {function_args}")
                    print(f"Function response: {function_response}")
                    print('-'*20)

            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            final_response = second_response.choices[0].message.content
        else:
            response = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            temperature=0.5,
            max_tokens=4096,
            top_p=1,
            stop=stop_s,
            stream=False,
            )
            final_response = response.choices[0].message.content
    else:
        response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        temperature=0.5,
        max_tokens=4096,
        top_p=1,
        stop=stop_s,
        stream=False,
        )
        #messages.append(response_message)
        final_response = response.choices[0].message.content
    
    return final_response

tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "search in the internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Responding a detail chat",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "serp_search",
            "description": "search in the internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Responding a detail chat",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "find recent news",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Responding a detail chat",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

if __name__ == "__main__":
    # Print the completion returned by the LLM.
    print(groq_llama3('You are a function calling LLM that uses function to find information from the internet. Suggest the organized result to understand easily for user. ', 'search about korea news', functioncall=True, verbose=True))
# %%
