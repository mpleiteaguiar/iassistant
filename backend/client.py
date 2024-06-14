from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import json

load_dotenv()

def sendQuestionGenIA(message,inputDocument,number_response_options,creativity_degree):

    seed_random_state = 999;
    max_list_options = number_response_options
    max_words = 1200
    #random variation of responses
    creativity = creativity_degree 
    openai_endpoint_images = "https://api.openai.com/v1/images/generations"
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_AGENT_ID=os.getenv("OPENAI_AGENT_ID")
    OPENAI_INSTRUCTIONS = os.getenv("OPENAI_INSTRUCTIONS").format(max_list_options)
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")

    client = OpenAI(api_key=OPENAI_API_KEY)
    client.api_key=OPENAI_API_KEY

    completion = client.chat.completions.create(
        max_tokens=max_words,
        model=OPENAI_MODEL,
        temperature=creativity,
        messages=[
          {"role": "system", "content": OPENAI_INSTRUCTIONS},
          {"role": "user", "content": message}
        ]
    )

    response = completion.choices[0].message.content;

    return response
    
if __name__=="__main__":
    print("="*20,"\nLoaded IAssistant Module!!!\n","="*20)