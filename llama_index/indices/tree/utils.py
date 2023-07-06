from typing import List, Optional

from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.schema import BaseNode


def get_numbered_text_from_nodes(
    node_list: List[BaseNode],
    text_splitter: Optional[TokenTextSplitter] = None,
) -> str:
    """Get text from nodes in the format of a numbered list.

    Used by tree-structured indices.

    """
    results = []
    for number, node in enumerate(node_list, start=1):
        node_text = " ".join(node.get_content().splitlines())
        if text_splitter is not None:
            node_text = text_splitter.truncate_text(node_text)
        text = f"({number}) {node_text}"
        results.append(text)
    return "\n\n".join(results)
