import streamlit as st
import pandas as pd

# Custom CSS for styled buttons
st.markdown("""
<style>
/* Main button styling */
.stButton > button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    padding: 12px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 200px;
}

.stButton > button:hover {
    transform: scale(1.1);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
}

.stButton > button:active {
    transform: scale(0.95);
}

/* Specific button styling */
.stButton > button[kind="primary"] {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
}

/* Update button styling */
.stButton > button[data-testid="baseButton-secondary"] {
    background: linear-gradient(45deg, #4ecdc4 0%, #44a08d 100%);
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(45deg, #44a08d 0%, #4ecdc4 100%);
}

/* Delete button styling */
.stButton > button[data-testid="baseButton-secondary"]:last-of-type {
    background: linear-gradient(45deg, #ff6b6b 0%, #ee5a24 100%);
}

.stButton > button[data-testid="baseButton-secondary"]:last-of-type:hover {
    background: linear-gradient(45deg, #ee5a24 0%, #ff6b6b 100%);
}

/* Center buttons */
.button-container {
    text-align: center;
    margin: 10px 0;
}

/* Add some spacing */
.stTextInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 10px;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
}

/* Style the number input */
.stNumberInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 10px;
    transition: all 0.3s ease;
}

.stNumberInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Task"])

st.title("Todo List")

#CREATE NEW TASK
new_task = st.text_input("Add new task")

# Custom styled add button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("➕ Add Task", key="add_task", type="primary"):
        if new_task.strip():
            new_row = pd.DataFrame({"Task": [new_task]})
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.rerun()

#DISPLAY TASKS
st.write("Tasks:")
st.dataframe(st.session_state.data)

#UPDATE TASK
if len(st.session_state.data) > 0:
    edit_index = st.number_input("Edit task", min_value=0, max_value=len(st.session_state.data)-1, step=1)
    edit_task = st.text_input("New Task Text")
    
    # Custom styled update button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✏️ Update Task", key="update_task"):
            if edit_task.strip():
                st.session_state.data.at[edit_index, "Task"] = edit_task
                st.rerun()

    #DELETE TASK
    delete_index = st.number_input("Delete task at row", min_value=0, max_value=len(st.session_state.data)-1, step=1)
    
    # Custom styled delete button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Delete Task", key="delete_task"):
            st.session_state.data = st.session_state.data.drop(index=delete_index).reset_index(drop=True)
            st.rerun()
else:
    st.write("No tasks to edit or delete. Add some tasks first!")
