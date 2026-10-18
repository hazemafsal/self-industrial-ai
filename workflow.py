
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from rag_engine import (
    retrieve_maintenance_context,
    generate_grounded_answer,
)


class MaintenanceState(TypedDict, total=False):
    question: str
    machine_context: str
    vectorstore: Any
    llm: Any
    documents: list
    answer: str


def build_workflow():
    def retrieve_node(state: MaintenanceState):
        documents = retrieve_maintenance_context(
            question=state["question"],
            machine_context=state["machine_context"],
            vectorstore=state["vectorstore"],
            k=4,
        )

        return {"documents": documents}

    def answer_node(state: MaintenanceState):
        answer = generate_grounded_answer(
            question=state["question"],
            machine_context=state["machine_context"],
            documents=state.get("documents", []),
            llm=state["llm"],
        )

        return {"answer": answer}

    graph = StateGraph(MaintenanceState)

    graph.add_node("retrieve_evidence", retrieve_node)
    graph.add_node("generate_explanation", answer_node)

    graph.add_edge(START, "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_explanation")
    graph.add_edge("generate_explanation", END)

    return graph.compile()


def run_maintenance_workflow(
    question,
    machine_context,
    vectorstore,
    llm,
):
    workflow = build_workflow()

    result = workflow.invoke({
        "question": question,
        "machine_context": machine_context,
        "vectorstore": vectorstore,
        "llm": llm,
    })

    return result
