from langchain_mistralai import ChatMistralAI
 
llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key="Ifc24iBKYhuLFBIhCZ9vkl12EOyp4Qdo",
    temperature=0
)
 
response = llm.invoke("What is Amazon EC2?")
 
print(response.content)