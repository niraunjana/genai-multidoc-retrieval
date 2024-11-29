## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
As the volume of research papers increases, efficiently retrieving and synthesizing information from multiple documents remains a challenge. This project aims to address the problem by creating a system that uses the LlamaIndex framework to retrieve and synthesize data from various academic papers. The system is designed to respond accurately to complex queries and provide insightful comparisons of the extracted information.

### DESIGN STEPS:

#### STEP 1: Dataset Preparation
- Downloaded research papers in PDF format, such as MetaGPT, SWE-Bench, LongLoRA, and LoftQ.
- Mapped each paper to its respective retrieval and summarization tools using a utility function (get_doc_tools).
#### STEP 2: Tool Generation
- Created vector tools for document similarity analysis and summary tools for abstracting key content.
- Stored these tools for each document in a dictionary (paper_to_tools_dict).

#### STEP 3: Index Construction
- Used VectorStoreIndex and ObjectIndex from LlamaIndex to index the document tools.
- Configured a retriever to fetch the most relevant tools based on similarity to the query.

#### STEP 4: Query Handling
- Developed an AgentWorker to process queries and retrieve necessary tools using the configured retriever.
- Implemented a FunctionCallingAgentWorker to ensure all responses are tool-based without reliance on prior knowledge.
  
#### STEP 5: Evaluation
- Designed diverse queries for testing, such as comparing evaluation datasets used in MetaGPT and SWE-Bench, and analyzing approaches in LongLoRA and LoftQ.
- Synthesized concise and comparative responses using the agent system.

### PROGRAM:
```py
from helper import get_openai_api_key
from utils import get_doc_tools
from pathlib import Path
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# Initialize API key and dependencies
nest_asyncio.apply()
OPENAI_API_KEY = get_openai_api_key()
llm = OpenAI(model="gpt-3.5-turbo")

# Define paper URLs and local paths
urls = [<list_of_urls>]
papers = [<list_of_local_files>]

# Generate tools for each paper
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# Create a retriever from indexed tools
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# Build agent with FunctionCallingAgentWorker
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt="""You are an agent designed to answer queries over a set of given papers. Use only the provided tools.""",
    verbose=True
)
agent = AgentRunner(agent_worker)

# Perform queries
response_1 = agent.query("Tell me about the evaluation dataset used in MetaGPT and compare it against SWE-Bench")
print(str(response_1))

response_2 = agent.query("Compare and contrast the LoRA papers (LongLoRA, LoftQ). Analyze the approach in each paper first.")
print(str(response_2))

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/da5e18bc-429e-4273-a44b-6c1927a985d5)


### RESULT:
The multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The agent demonstrated its capability to retrieve and synthesize information from multiple academic papers, answering complex queries with concise, relevant, and accurate responses.
