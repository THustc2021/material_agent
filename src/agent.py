from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(llm, tools, prompt_content: str,system_message: str):
    """Create an agent."""
    tools_name = [tool.name for tool in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f'{prompt_content}. You have the following tools available: {tools_name}.',
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)