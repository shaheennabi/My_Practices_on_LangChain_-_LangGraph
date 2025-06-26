from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
    })

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
