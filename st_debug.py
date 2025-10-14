"""
Streamlit debug utilities for the Legal Drafting System.
Provides real-time debug logging in the Streamlit interface.
"""

import streamlit as st
import datetime
from inspect import getframeinfo, stack


def debug(message: str) -> None:
    """
    Add a debug message to the Streamlit debug window.
    
    Args:
        message (str): Debug message to display
    """
    # Initialize debug string if not exists
    if "debug_string" not in st.session_state:
        st.session_state["debug_string"] = "<b>Debug window</b>"
    
    # Get current time and caller information
    now = datetime.datetime.now()
    caller = getframeinfo(stack()[1][0])
    
    # Format debug entry with timestamp, file location, and message
    entry = (
        f"<div style='border-bottom: dotted; border-width: thin; border-color: #cccccc;'>"
        f"{now.strftime('%H:%M:%S')} [{caller.filename}:{caller.lineno}] {message}</div>"
    )
    
    # Prepend new entry to debug string (most recent first)
    st.session_state["debug_string"] = entry + st.session_state["debug_string"]


