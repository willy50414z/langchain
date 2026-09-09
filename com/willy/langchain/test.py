from langgraph import phase2_wf

config = {"configurable":{"thread_id":"session1"}}
# req = "LangChain_Professional_Learning_Map.md"
#
graph = phase2_wf.build()
# for event in graph.stream({"requirement":req}, config):
#     print(event)

# # 2. 檢視當前狀態（人類審核階段）🔍
# current_state = graph.get_state(config)
# print("目前暫停在：", current_state.next)
# print("目前的計畫：", current_state.values.get("plan"))
#
# # 3. 人類確認後，傳入 None 恢復執行 ▶️
# for event in graph.stream(None, config):
#     print(event)

current_state = graph.get_state(config)
print("目前暫停在：", current_state.next)
print("目前的計畫：", current_state.values.get("plan"))