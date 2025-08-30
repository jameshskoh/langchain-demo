from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    model = init_chat_model(
        "qwen/qwen3-30b-a3b-2507",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="dummy",
    )

    messages = [
        SystemMessage(content="This chat is mainly about ice-breaking activities."),
        HumanMessage(content="I want an activity that challenges the mind."),
    ]

    # direct
    # model.invoke(messages)

    # streaming
    for token in model.stream(messages):
        print(token.content, end="")
