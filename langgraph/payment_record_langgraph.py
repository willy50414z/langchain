import sys
from typing import TypedDict, List

from langgraph.graph import StateGraph
from langgraph.types import interrupt

START = sys.intern("__start__")
END = sys.intern("__end__")


class PaymentRecordState(TypedDict):
    total_payment: int
    total_pay_times: int
    buy_things: List[str]
    account_balance: int
    wallet_enough: bool


def erase_default(state: PaymentRecordState) -> PaymentRecordState:
    state.setdefault("total_payment", 0)
    state.setdefault("total_pay_times", 0)
    state.setdefault("buy_things", [])
    state.setdefault("account_balance", 100)
    state.setdefault("wallet_enough", True)
    return state


def breakfast(state: PaymentRecordState) -> PaymentRecordState:
    erase_default(state)

    state["total_payment"] += 50
    state["total_pay_times"] += 1
    state["buy_things"].append("早餐")
    return state


def dinner(state: PaymentRecordState) -> PaymentRecordState:
    erase_default(state)

    state["total_payment"] += 150
    state["total_pay_times"] += 1
    state["buy_things"].append("晚餐")
    return state


def other_payment(state: PaymentRecordState) -> PaymentRecordState:
    erase_default(state)

    other_payments = interrupt({"requests": "請輸入今日消費清單", "格式": "{'便當':99,'飲料':88}"})

    for item in other_payments.keys():
        state["total_payment"] += other_payments[item]
        state["total_pay_times"] += 1
        state["buy_things"].append(item)

    return state


def check_wallet(state: PaymentRecordState) -> PaymentRecordState:
    erase_default(state)

    if state["total_payment"] > state["account_balance"]:
        add_account = interrupt({"request": "沒錢了，請存錢"})
        state["account_balance"] += add_account

    if state["account_balance"] < state["total_payment"]:
        state["wallet_enough"] = False
    else:
        state["wallet_enough"] = True
    return state


# checkpointer = MemorySaver()

workflow = StateGraph(PaymentRecordState)

workflow.add_node("breakfast", breakfast)
workflow.add_node("dinner", dinner)
workflow.add_node("other_payment", other_payment)
workflow.add_node("check_wallet", check_wallet)

workflow.add_edge(START, "breakfast")
workflow.add_edge("breakfast", "dinner")
workflow.add_edge("dinner", "other_payment")
workflow.add_edge("other_payment", "check_wallet")
workflow.add_conditional_edges("check_wallet", lambda state: state["wallet_enough"], {True: "breakfast", False: END})

graph = workflow.compile()
