import streamlit as st
from main import get_answer


st.set_page_config(page_title="FinSight")

st.title("FinSight")

st.subheader("Ask about your financial report")

col1, col2 = st.columns([5,1])

with col1:
    query = st.text_input("Ask a question about the financial report", label_visibility="collapsed")

with col2:
    search = st.button("Search")

if search and query.strip() != "":

    with st.spinner("Analyzing documents..."):

        answer, contexts = get_answer(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, (text, page) in enumerate(contexts):

        st.markdown(f"**Source {i+1} — Page {page}**")

        st.write(text)