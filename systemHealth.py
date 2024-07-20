import nest_asyncio
import os
from flask import Flask, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.agent.openai import OpenAIAssistantAgent
# CMDB-related setup
from langchain_community.chat_message_histories import ChatMessageHistory

nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)

# Finance-related setup
documents = SimpleDirectoryReader(r"/Users/aparajitayadav/Desktop/BhaktiGPT/Data").load_data()
os.environ["OPENAI_API_KEY"] = ""

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
    description="Useful for summarizing HTTP logs to understand the system health. You provide detailed oriented reports which shows number of failures and remediation steps for that"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_tool",
    description="Useful for retrieving specific context to answer specific questions about the finance statement"
)

agent = OpenAIAssistantAgent.from_new(
    name="QA bot",
    instructions="You are an amazing System Analysts who takes care of system health by checking, Aggregating and analyzing logs and providing summary daily related to instance health",
    openai_tools=[],
    tools=[summary_tool, vector_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0
)

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


if __name__ == '__main__':
    app.run(debug=True)
