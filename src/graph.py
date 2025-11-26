import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Union, TypedDict

from config.load_config import WORKING_DIRECTORY, config, save_dialogure
from src.tools import *
from src.prompt import dft_agent_prompt, hpc_agent_prompt, supervisor_prompt, members


class myStep(BaseModel):
    """Step in the plan."""
    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )


class PlanExecute(TypedDict):
    input: str
    plan: List[myStep]
    past_steps: List[myStep]
    response: str
    next: str


class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[myStep] = Field(
        description=f"""
        Steps to follow in future. Each step is a tuple of (step, agent). agent can only be chosen from {members}.
        """
        # description="different steps to follow, should be in sorted order"
        # description="""different steps to follow (first element of the Tuple), and the agent in charge for each step (second element of the Tuple),
        # should be in sorted order by the order of execution"""
    )


class Response(BaseModel):
    """End everything and response to the user."""
    response: str


class Act(BaseModel):
    """Action to perform."""
    action: Union[Plan, Response] = Field(
        description="Action to perform. If you need to further use tools to get the answer, use Plan."
                    "If you want to end the conversation, use Response."
    )


def print_stream(s):
    if "messages" not in s:
        print("#################")
        if save_dialogure:
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write("#################\n")
        print(s)
        if save_dialogure:
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(repr(s))
                f.write("\n")
    else:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
            if save_dialogure:
                with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                    f.write(repr(message))
                    f.write("\n")
        else:
            if hasattr(message, 'usage_metadata'):
                print(
                    f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}")
                if save_dialogure:
                    with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                        f.write(
                            f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}\n")
            message.pretty_print()
            if save_dialogure:
                with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                    f.write(message.pretty_repr())
                    f.write("\n")
    print()
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write("\n")


def supervisor_chain_node(state, chain, name):
    # read "status.txt" in the working directory
    with open(f"{WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, supervisor is waiting. cwd: {WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()

    print(f"supervisor is processing!!!!!")
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"supervisor is processing!!!!!\n")

    print(state)
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(str(state))
            f.write("\n")

    output = chain.invoke(state)

    if isinstance(output.action, Response):
        return {"response": output.action.response, "next": "FINISH"}
    else:
        if len(output.action.steps) == 0:
            return {"plan": output.action.steps, "next": name}
        else:
            return {"plan": output.action.steps, "next": output.action.steps[0].agent}


def worker_agent_node(state, agent, name, past_steps_list):
    # read "status.txt" in the working directory
    with open(f"{WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, {name} Agent is waiting. cwd: {WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()

    print(f"Agent {name} is processing!!!!!")
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"Agent {name} is processing!!!!!\n")

    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step.step}" for i, step in enumerate(plan))
    print(plan_str)
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(plan_str)
            f.write("\n")
    task = plan[0]
    #     task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i + 1}. {step.agent}: {step.step}" for i, step in enumerate(past_steps_list))
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}
Here is the overall objective:
{state["input"]}
Now, you are tasked with: {task}. 
If a tool is relevant for this task, you MUST call it using the proper tool calling protocol.
Do NOT return JSON. Do NOT describe the tool. Directly invoke the tool with tool call.
"""

    print(task_formatted)
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(task_formatted)
            f.write("\n")
    print(f"Agent {name} is processing!!!!!")
    if save_dialogure:
        with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"Agent {name} is processing!!!!!\n")

    for agent_response in agent.stream(
            {"messages": [{"role": "user", "content": task_formatted}]},
            {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    ):
        # set agent_response to be the value of the first key of the dictionary
        agent_response_p = next(iter(agent_response.values()))

        past_steps_list.append(myStep(step=agent_response_p["messages"][-1].content, agent=name))

        print_stream(agent_response_p)

    return {
        "past_steps": past_steps_list,
    }


def create_planning_graph() -> CompiledStateGraph:
    # create a file named status.txt in the working directory
    with open(f"{WORKING_DIRECTORY}/status.txt", "w") as f:
        f.write("run")

    # Define the model
    llm = create_chat_model(config, temperature=0.5)
    workerllm = create_chat_model(config, temperature=0.5)

    # System Supervisor with tool bind and with_structured_output
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
    ]
    supervisor_chain = ChatPromptTemplate.from_template(supervisor_prompt) | llm.bind_tools(supervisor_tools).with_structured_output(Act)
    supervisor_agent = functools.partial(supervisor_chain_node, chain=supervisor_chain, name="Supervisor")

    PAST_STEPS = []

    ### DFT Agent
    dft_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        calculate_formation_E,
        generateSurface_and_getPossibleSite,
        generate_myAdsorbate,
        add_myAdsorbate,
        init_structure_data,
        find_pseudopotential,
        write_QE_script_w_ASE,
        calculate_lc,
        generate_convergence_test,
        get_kspacing_ecutwfc,
        generate_eos_test,
        read_energy_from_output,
        get_convergence_suggestions,
        analyze_BEEF_result
    ]
    dft_agent = create_agent(workerllm, tools=dft_tools, system_prompt=dft_agent_prompt)
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent", past_steps_list=PAST_STEPS)

    ### HPC Agent
    hpc_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        submit_and_monitor_job,
        add_resource_suggestion
    ]
    hpc_agent = create_agent(workerllm, tools=hpc_tools, system_prompt=hpc_agent_prompt)
    hpc_node = functools.partial(worker_agent_node, agent=hpc_agent, name="HPC_Agent", past_steps_list=PAST_STEPS)

    # Create the graph
    graph = StateGraph(PlanExecute)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
    graph.add_node("Supervisor", supervisor_agent)

    # We want our workers to ALWAYS "report back" to the supervisor when done
    for member in members:
        graph.add_edge(member, "Supervisor")

    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    conditional_map["Supervisor"] = "Supervisor"
    graph.add_conditional_edges("Supervisor", lambda state: state["next"], conditional_map)
    graph.add_edge(START, "Supervisor")

    return graph.compile()