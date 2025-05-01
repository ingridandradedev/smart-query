# Smart Query - ReAct Agent for Marketing Data Analysis

![Smart Query Data](https://offgridmartech.com.br/ai-microsoft-hackathon/smart_query_data.png)

Smart Query is a custom Reasoning and Action (ReAct) agent built using LangGraph. This agent assists with marketing data analysis by leveraging tools for querying databases, analyzing knowledge bases, and providing actionable insights. The project is highly extensible and adaptable to various use cases.

## Features
- **ReAct Agent**: Implements a reasoning and action loop to process user queries and execute actions iteratively.
- **Database Integration**: Tools for listing tables, retrieving table columns, and executing read-only SQL queries on PostgreSQL databases.
- **Knowledge Base Search**: Integration with Pinecone and OpenAI for querying knowledge bases using embeddings.
- **Customizable Prompts**: Configurable system prompts to define the agent's behavior and context.
- **Tool Integration**: Includes tools for web search, database operations, and knowledge base queries.
- **FastAPI Interface**: Provides an API for interacting with the agent, supporting both synchronous and streaming responses.

## How It Works
The ReAct agent follows a structured process:
1. **User Query**: The user provides a query or request.
2. **Reasoning**: The agent reasons about the query and decides on an action.
3. **Action Execution**: The agent executes the chosen action using integrated tools.
4. **Observation**: The agent observes the result of the action.
5. **Iteration**: Steps 2-4 are repeated until the agent can provide a final response.

## Setup

### User Interface

1. **Login**

Watch the demo video showcasing the Smart Query user interface and features:

<video width="640" height="360" controls>
  <source src="https://offgridmartech.com.br/ai-microsoft-hackathon/smart_query_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Prerequisites
- Python 3.9 or higher
- PostgreSQL database (for database tools)
- API keys for Tavily, OpenAI, and Pinecone (if using related tools)

### Installation
1. Clone the repository:
  ```bash
  git clone https://github.com/your-repo/smart-query.git
  cd smart-query
  ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Create a `.env` file based on `.env.example` and add your API keys.
4. Run the application.

## API Endpoints

### `/invoke`
- **Method**: POST  
- **Description**: Processes a user query and returns the agent's response.  
- **Request Body**:
  ```json
  {
   "messages": [
    {"role": "user", "content": "What are the top-performing campaigns?"}
   ],
   "user_id": "123",
   "user_name": "John Doe",
   "database_schema": "marketing"
  }
  ```
- **Response**:
  ```json
  {
   "thread_id": "abc123",
   "user_id": "123",
   "user_name": "John Doe",
   "messages": [
    {"role": "assistant", "content": "The top-performing campaigns are..."}
   ]
  }
  ```

### `/invoke_last`
- **Method**: POST  
- **Description**: Returns only the last message from the agent's response.

### `/stream`
- **Method**: POST  
- **Description**: Streams the agent's response in real-time.

## Customization

### Add New Tools
Extend the agent's capabilities by adding new tools in `tools.py`. For example, you can add tools for additional database operations or external API integrations.

### Modify the Prompt
The system prompt is defined in `prompts.py`. Customize it to change the agent's behavior and context.

### Change the Model
The default model is `anthropic/claude-3-5-sonnet-20240620`. Switch to other models by updating the `model` field in `configuration.py`.

## Development

### Run Tests
To run unit tests:
```bash
make test
```

### Lint and Format Code
To lint and format the code:
```bash
make lint
make format
```

### Docker Support
Build and run the application using Docker:
```bash
docker build -t smart-query .
docker run -p 8000:8000 smart-query
```

## Folder Structure
- **`react_agent`**: Core logic for the ReAct agent, including tools, configuration, and state management.
- **`.langgraph_api`**: Checkpoints and data for LangGraph.
- **`tests`**: Unit tests for the application.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

For more details, refer to the LangGraph documentation.
