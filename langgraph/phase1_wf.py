import operator
import sys
from typing import TypedDict, List, Annotated

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from urllib3.util import timeout


# 1.建立state
class CodeDevState(TypedDict):
    status: str
    requirement: str
    messages: Annotated[List[BaseMessage], operator.add]


# 3.設定tools
@tool
def write_file(file_path, content):
    """將指定內容寫入本地檔案系統中的檔案。"""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except IOError:
        print("File not exist")
        return False


LLM_TOOLS = [write_file]

# 4.設定llm
llm = ChatOllama(base_url="https://ollama.cwh0628.qzz.io", model="gemma4", temperature=0.2)
llm_with_tools = llm.bind_tools(LLM_TOOLS)


# 5. 建立nodes
def implement(state: CodeDevState):
    sys_msg = SystemMessage(content=(
        "你是一名 Senior Developer。請根據需求撰寫程式碼，"
        "並且 **必須呼叫 write_file 工具** 將程式碼存入檔案中。"
    ))
    user_msg = HumanMessage(content=state["requirement"])
    res = llm_with_tools.invoke([sys_msg, user_msg])
    return {"messages": [res]}

# 建立graph
START = sys.intern("__start__")
END = sys.intern("__end__")
workflow = StateGraph(CodeDevState)

# add nodes
workflow.add_node("implement", implement)
workflow.add_node("tools", ToolNode(LLM_TOOLS))

workflow.set_entry_point("implement")
# 如果符合條件就呼叫tool
workflow.add_conditional_edges("implement", tools_condition, {
    "tools": "tools",
    "__end__": END
})
workflow.add_edge("tools",END)
graph = workflow.compile()
