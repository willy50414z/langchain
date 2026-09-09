import operator
from typing import List, Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class DevState(TypedDict):
    step: str
    messages: Annotated[List[BaseMessage], operator.add]
    requirement: str
    plan: str
    origin_plan: str
    user_advice: str


# 1. 註冊tool
@tool
def write_file(file_path: str, content: str):
    """將文字內容匯出到本地指定路徑。

    Args:
        file_path (str): 要寫入的檔案路徑或檔名，例如 'learning_map.md' 或 'output.txt'。
        content (str): 要寫入檔案的完整文字內容。
    """
    try:
        with open(file_path, "w") as file:
            file.write(content)
    except:
        return False
    return True


llm = ChatOllama(base_url="http://localhost:11434", model="gemma4:e2b", temperature=0.2)
binded_tool_llm = llm.bind_tools([write_file])


# create node
def plan(dev_state: DevState):
    sys_msg = SystemMessage(content="你是一個專業顧問，你可以根據需求拆解任務，並制定出最合適的執行計畫")
    origin_plan = None
    user_advice = dev_state.get("user_advice")
    if user_advice:
        origin_plan = dev_state["plan"]
        llm_res = llm.invoke([sys_msg, HumanMessage(
            content=f'user requirement:{dev_state["requirement"]}]. Our origin plan:[{origin_plan}]. '
                    f'Below is user advice for this plan:[{user_advice}]. '
                    f'Please refer to user''s advice and make a new plan')])
    else:
        llm_res = llm.invoke([sys_msg, HumanMessage(content=dev_state["requirement"])])
    return {"plan": llm_res.content, 'origin_plan': origin_plan}


def review_plan(dev_state: DevState):
    sys_msg = SystemMessage(
        content="你是一個嚴格的計畫審查專家，需要根據需求審查目前的執行計畫是否合理，修正後提出一份優化過的執行計畫")
    user_advice = dev_state.get("user_advice")
    if user_advice:
        llm_res = llm.invoke(
            [sys_msg,
             HumanMessage(content=f'User has some advice for our old plan. I have make a new plan.'
                                  f'Please help to improved our new plan'
                                  f'User requirement:{dev_state["requirement"]}].'
                                  f'Old plan:[{dev_state["origin_plan"]}].'
                                  f'User advice for old plan:[{user_advice}]. '
                                  f'Our new plan:[{dev_state["plan"]}]')])
    else:
        llm_res = llm.invoke(
            [sys_msg,
             HumanMessage(content=f'user的需求:[{dev_state["requirement"]}]，目前第一版的執行計畫{dev_state["plan"]}')])
    return {"plan": llm_res.content, 'user_advice':None}


def implement(state: DevState):
    sys_msg = SystemMessage(
        content="你是一個資深開發人員,可以根據制定好的執行計畫執行，並在 **需要匯出文字檔案時呼叫write_file工具**")
    llm_res = binded_tool_llm.invoke([sys_msg, HumanMessage(
        content=f"這是原始需求:[{state['requirement']}]，這是經過review且approve的執行計畫{state['plan']}，請依據執行計畫完成user需求")])
    return {"messages": [llm_res]}


def replan_condition(state: DevState):
    if state['user_advice']:
        return 'replan'
    else:
        return 'approve'


def build():
    memory = MemorySaver()

    workflow = StateGraph(DevState)

    workflow.add_node("plan", plan)
    workflow.add_node("review_plan", review_plan)
    workflow.add_node("implement", implement)

    tool_node = ToolNode([write_file])
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "review_plan")
    workflow.add_conditional_edges("review_plan", replan_condition, {
        'replan': "plan"
        , 'approve': 'implement'
    })
    workflow.add_conditional_edges("implement", tools_condition, {
        "tools": "tools"
        , "__end__": END
    })
    workflow.add_edge("tools", "__end__")

    return workflow.compile(checkpointer=memory, interrupt_after=["review_plan"])


graph = build()
