import streamlit as st
import datetime
from inspect import getframeinfo, stack


def debug(message: str) -> None:
    if "debug_string" not in st.session_state:
        st.session_state["debug_string"] = "<b>Debug window</b>"
    now = datetime.datetime.now()
    caller = getframeinfo(stack()[1][0])
    entry = (
        f"<div style='border-bottom: dotted; border-width: thin; border-color: #cccccc;'>"
        f"{now.strftime('%H:%M:%S')} [{caller.filename}:{caller.lineno}] {message}</div>"
    )
    st.session_state["debug_string"] = entry + st.session_state["debug_string"]


