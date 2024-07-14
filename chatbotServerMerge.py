import nest_asyncio
import os
from flask import Flask, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.agent.openai import OpenAIAssistantAgent
# CMDB-related setup
import pandas as pd
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core import PromptTemplate
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)

# Finance-related setup
documents = SimpleDirectoryReader(r"C:\Users\isikc\Downloads\ChatBot\data").load_data()
os.environ["OPENAI_API_KEY"] 

Settings.llm = OpenAI()
Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
vector_query_engine = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description="Useful for summarization questions related to the finance analysis"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_tool",
    description="Useful for retrieving specific context to answer specific questions about the finance statement"
)

agent = OpenAIAssistantAgent.from_new(
    name="QA bot",
    instructions="You are a financial analyst designed to answer questions about the financial statements",
    openai_tools=[],
    tools=[summary_tool, vector_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0
)



instruction_str = "Provide a detailed description of the dataset."
df = pd.read_csv('cmdb_ci.csv')  # Adjust the path to your CSV file

pandas_prompt_str = (
    "Given the following dataframe, perform the requested operation:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(instruction_str=instruction_str, df_str=df.head(5))
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = OpenAI(model="gpt-3.5-turbo-0125")

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
    ]
)
qp.add_link("response_synthesis_prompt", "llm2")


@app.route('/finance', methods=['POST'])
def finance_query():
    data = request.json
    query_str = data.get('query', '')
    print(query_str)

    if not query_str:
        return jsonify({'error': 'Query is required'}), 400

    try:
        response = agent.chat(query_str)
        print(response)
        return jsonify({'response': response.response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def general_query():
    data = request.json
    query_str = data.get('query', '')
    print(query_str)
    
    if not query_str:
        return jsonify({'error': 'Query is required'}), 400

    try:
        response = qp.run(query_str=query_str)
        return jsonify({'response': response.message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
